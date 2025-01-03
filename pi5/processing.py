# processing.py
#
# This module handles the core frame processing pipeline, including:
# - Frame capture from the camera
# - Motion detection and analysis
# - Mask inference and tracking
# - Effect application
#
# The processing pipeline is split into separate threads to maximize performance
# on multi-core systems while maintaining real-time processing capabilities.

# processing.py

import cv2 
import numpy as np
import time
import traceback
from picamera2 import Picamera2

from utils import (
    perf_stats,
    get_cpu_temp,
    log_performance_stats,
    update_performance_stats
)
from motion import detect_motion
from mask import update_mask_tracking, create_physics_mask
import shared_state as state
from effects import apply_effect

def capture_frames():
    """
    Continuously capture frames from the camera and update the current frame in shared state.
    This function runs in its own thread and handles all camera interaction.
    """
    last_capture_time = time.time()
    
    try:
        # Initialize camera with optimal settings for our use case
        state.picam2 = Picamera2()
        preview_config = state.picam2.create_preview_configuration(
            main={"size": state.INPUT_SIZE},
            buffer_count=4
        )
        state.picam2.configure(preview_config)
        state.picam2.start()

        while True:
            try:
                frame = state.picam2.capture_array("main")
                current_time = time.time()
                perf_stats['frame_intervals'].append(current_time - last_capture_time)
                last_capture_time = current_time
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if state.camera_flipped_h:
                    frame = cv2.flip(frame, 1)
                if state.camera_flipped_v:
                    frame = cv2.flip(frame, 0)
                
                with state.frame_lock:
                    if state.current_frame is not None:
                        state.previous_frame = state.current_frame.copy()
                    state.current_frame = frame.copy()
                
                time.sleep(max(0, (1/60) - (time.time() - current_time)))
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal capture error: {e}")
        traceback.print_exc()
    finally:
        if state.picam2 is not None:
            state.picam2.stop()

def separate_objects(masks, scores, min_size=100, min_confidence=0.5, iou_threshold=0.5):
    """
    Separate individual objects from SAM's mask proposals.
    
    Args:
        masks: Array of mask proposals from SAM (N, H, W)
        scores: Confidence scores for each mask
        min_size: Minimum object size in pixels
        min_confidence: Minimum confidence score to consider
        iou_threshold: Threshold for considering masks as the same object
    
    Returns:
        List of unique object masks
    """
    if masks is None or len(masks) == 0:
        return []
        
    # Get initial mask dimensions
    N = masks.shape[0]
    object_masks = []
    used_masks = set()
    
    # Sort masks by score for priority
    mask_indices = np.argsort(scores[0])[::-1]
    
    for i in mask_indices:
        if i in used_masks or scores[0][i] < min_confidence:
            continue
            
        current_mask = masks[0][i]
        
        # Check if mask is big enough
        if np.sum(current_mask) < min_size:
            continue
            
        # Check if this mask significantly overlaps with any existing object
        is_unique = True
        for existing_mask in object_masks:
            intersection = np.logical_and(current_mask, existing_mask).sum()
            union = np.logical_or(current_mask, existing_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                is_unique = False
                break
                
        if is_unique:
            object_masks.append(current_mask)
            used_masks.add(i)
            
    return object_masks

def inference_loop(sam):
    """Run continuous mask inference with enhanced debugging."""
    print("Starting inference loop with debug logging")
    h, w = (120, 213)
    
    # Create grid points (6x6 grid = 36 points)
    points_per_side = 6
    x = np.linspace(0, w, points_per_side)
    y = np.linspace(0, h, points_per_side)
    xv, yv = np.meshgrid(x, y)
    fixed_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    
    # ONNX inputs setup
    onnx_coord = np.concatenate([fixed_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([np.ones(len(fixed_points)), np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    target_frame_time = 1/20
    last_inference_time = time.time()
    frame_counter = 0

    while True:
        try:
            current_time = time.time()
            if current_time - last_inference_time < target_frame_time:
                time.sleep(target_frame_time - (current_time - last_inference_time))
                continue
                
            frame_counter += 1
            if frame_counter % 30 == 0:  # Log every 30 frames
                print(f"\n=== Frame {frame_counter} ===")
            
            with state.frame_lock:
                if state.current_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_process = state.current_frame.copy()
                print(f"Got frame: shape={frame_to_process.shape}")

            # Prepare frame
            frame_small = cv2.resize(frame_to_process, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Run SAM inference
            state.predictor.set_image(frame_rgb)
            state.image_embedding = state.predictor.get_image_embedding().cpu().numpy()

            onnx_coord_frame = state.predictor.transform.apply_coords(
                onnx_coord, 
                frame_small.shape[:2]
            ).astype(np.float32)
            
            ort_inputs = {
                "image_embeddings": state.image_embedding,
                "point_coords": onnx_coord_frame,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(frame_small.shape[:2], dtype=np.float32)
            }
            
            print("Running ONNX inference...")
            masks, scores, _ = state.ort_session.run(None, ort_inputs)
            masks = masks > (state.predictor.model.mask_threshold + state.MASK_THRESHOLD_ADJUSTMENT)
            
            print(f"Got masks: shape={masks.shape}, scores shape={scores.shape}")
            if masks is not None:
                print(f"Number of masks > 0.5 confidence: {np.sum(scores[0] > 0.5)}")

            # Find the best non-overlapping masks
            object_masks = []
            used_indices = set()
            
            # Sort masks by confidence
            mask_indices = np.argsort(scores[0])[::-1]
            
            for idx in mask_indices:
                if scores[0][idx] < 0.5:  # Skip low confidence masks
                    continue
                    
                current_mask = masks[0][idx]
                mask_area = np.sum(current_mask)
                
                if mask_area < 100:  # Skip tiny masks
                    continue
                    
                # Check if this mask overlaps significantly with existing masks
                is_unique = True
                for existing_mask in object_masks:
                    intersection = np.logical_and(current_mask, existing_mask).sum()
                    union = np.logical_or(current_mask, existing_mask).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5:  # If significant overlap
                        is_unique = False
                        break
                
                if is_unique:
                    object_masks.append(current_mask)
                    used_indices.add(idx)
                    
                    if len(object_masks) >= 5:  # Limit to 5 objects for performance
                        break
            
            print(f"Found {len(object_masks)} unique objects")

            if object_masks:
                frame_height, frame_width = frame_to_process.shape[:2]
                physics_masks = []
                
                # Process each unique object mask
                for i, object_mask in enumerate(object_masks):
                    # Resize mask to full frame size
                    mask_resized = cv2.resize(
                        object_mask.astype(np.uint8),
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Get mask properties
                    y_indices, x_indices = np.where(mask_resized)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        # Create physics mask
                        physics_mask = create_physics_mask(
                            mask_resized,
                            frame_to_process.shape[:2],
                            mass=1.0,
                            friction=0.95
                        )
                        physics_masks.append(physics_mask)
                
                print(f"Created {len(physics_masks)} physics masks")
                
                # Update shared state
                with state.frame_lock:
                    state.current_masks = physics_masks
                    state.mask_scores = [1.0] * len(physics_masks)
                    print(f"Updated shared state with {len(state.current_masks)} masks")
            else:
                with state.frame_lock:
                    state.current_masks = []
                    state.mask_scores = None
                    print("No masks found this frame")

            last_inference_time = time.time()

        except Exception as e:
            print(f"Error in inference loop: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)

def process_frames():
    """
    Process captured frames with motion detection and prepare them for visualization.
    This function runs in its own thread and handles the core frame processing pipeline.
    
    The processing steps include:
    1. Motion detection between consecutive frames
    2. Statistics collection and logging
    3. Frame processing for effects
    
    Processed frames are stored in shared state for the renderer to consume.
    """
    print("Starting process frames thread")
    
    last_process_time = time.time()
    target_interval = 1/30  # Target 30 FPS for processing

    print("Waiting for first frame...")
    while state.current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        while True:
            try:
                # Get the latest frame with thread safety
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                    frame_to_process = state.current_frame.copy()
                    previous_frame = state.previous_frame.copy() if state.previous_frame is not None else None

                # Track processing performance
                process_start_time = time.time()
                perf_stats['cpu_temps'].append(get_cpu_temp())
                
                # Rate limit processing
                current_time = time.time()
                time_since_last = current_time - last_process_time
                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue

                # Detect motion between frames
                motion_contours, debug_frame, motion_intensity = detect_motion(frame_to_process, previous_frame)
                
                # Update shared state with processing results
                with state.frame_lock:
                    # Use debug frame to see what's being detected
                    state.raw_processed_frame = debug_frame if debug_frame is not None else frame_to_process.copy()
                    state.motion_contours = motion_contours
                print("Frame processing complete")

                # Update timing statistics
                last_process_time = current_time
                perf_stats['total_process_times'].append(time.time() - process_start_time)
                
                # Log performance statistics
                log_performance_stats()

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        traceback.print_exc()

def apply_effects_loop():
    """
    Apply visual effects to processed frames using current masks.
    This function runs in its own thread and handles all effect processing.
    
    Effects are applied based on:
    1. The currently selected effect
    2. The current mask state
    3. Motion detection results
    
    The results are stored in shared state for the renderer to consume.
    """
    print("Starting effects loop")
    while True:
        current_time = time.time()
        try:
            if state.current_effect is not None:
                print(f"\nEffect Debug:")
                print(f"Current effect: {state.current_effect}")
                print(f"Has raw frame: {state.raw_processed_frame is not None}")
                print(f"Has masks: {state.current_masks is not None}")
                
                # Get current state with thread safety
                with state.frame_lock:
                    if state.current_masks is not None:
                        print(f"Masks shape: {state.current_masks.shape}")
                        print(f"Mask flags: {state.mask_flags}")
                    
                    if state.raw_processed_frame is not None:
                        frame_to_process = state.raw_processed_frame.copy()
                        masks_for_effect = state.current_masks
                        
                        # Apply the selected effect
                        result = apply_effect(frame_to_process, state.current_effect, 
                                           state.effect_params, masks_for_effect)
                        
                        # Store result in shared state
                        state.processed_frame = result
                    else:
                        print("No raw frame available")

            # Rate limit effect processing to 60 FPS
            time.sleep(max(0, (1/60) - (time.time() - current_time)))

        except Exception as e:
            print(f"Effects error: {e}")
            traceback.print_exc()
            time.sleep(0.1)
