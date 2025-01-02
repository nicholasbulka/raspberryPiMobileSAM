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

import cv2
import numpy as np
import time
import traceback
from picamera2 import Picamera2

from utils import (
    perf_stats, INPUT_SIZE, POINTS_PER_SIDE,
    get_cpu_temp, log_performance_stats,
    update_mask_tracking, detect_motion
)
import shared_state as state
from effects import apply_effect

def capture_frames():
    """
    Continuously capture frames from the camera and update the current frame in shared state.
    This function runs in its own thread and handles all camera interaction.
    
    The captured frames are stored in shared state for access by other processing threads.
    Frame capture is rate-limited to avoid overwhelming the system.
    """
    last_capture_time = time.time()
    
    try:
        # Initialize camera with optimal settings for our use case
        state.picam2 = Picamera2()
        preview_config = state.picam2.create_preview_configuration(
            main={"size": INPUT_SIZE},
            buffer_count=4  # Increased buffer for smoother capture
        )
        state.picam2.configure(preview_config)
        state.picam2.start()

        while True:
            try:
                # Capture new frame from camera
                frame = state.picam2.capture_array("main")
                current_time = time.time()
                perf_stats['frame_intervals'].append(current_time - last_capture_time)
                last_capture_time = current_time
                
                # Convert to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply camera flips if enabled
                if state.camera_flipped_h:
                    frame = cv2.flip(frame, 1)
                if state.camera_flipped_v:
                    frame = cv2.flip(frame, 0)
                
                # Update shared state with thread safety
                with state.frame_lock:
                    if state.current_frame is not None:
                        state.previous_frame = state.current_frame.copy()
                    state.current_frame = frame.copy()
                
                # Rate limit to 60 FPS maximum
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
                motion_mask, motion_intensity = detect_motion(frame_to_process, previous_frame)
                
                # Update shared state with processing results
                with state.frame_lock:
                    state.raw_processed_frame = frame_to_process.copy()
                    state.motion_mask = motion_mask
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

def inference_loop(sam):
    """
    Run continuous mask inference on processed frames.
    This function runs in its own thread and handles all model inference.
    
    The inference process includes:
    1. Frame preprocessing for the model
    2. Running the SAM model
    3. Post-processing masks
    4. Updating mask tracking statistics
    
    Args:
        sam: The initialized MobileSAM model instance
    """
    # Configure inference parameters
    h, w = (180, 320)  # Reduced size for inference while keeping aspect ratio
    
    # Pre-calculate grid points for prompting
    points_per_side = POINTS_PER_SIDE
    x = np.linspace(0, w, points_per_side)
    y = np.linspace(0, h, points_per_side)
    xv, yv = np.meshgrid(x, y)
    fixed_points = np.stack([xv.flatten(), yv.flatten()], axis=1) 
    
    # Pre-allocate arrays for ONNX
    onnx_coord = np.concatenate([fixed_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([np.ones(len(fixed_points)), np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    while True:
        try:
            # Get the latest frame with thread safety
            with state.frame_lock:
                if state.current_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_process = state.current_frame.copy()

            # Preprocess frame
            t_resize_start = time.time()
            frame_small = cv2.resize(frame_to_process, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            perf_stats['resize_times'].append(time.time() - t_resize_start)

            # Get frame embedding
            t_embed_start = time.time()
            state.predictor.set_image(frame_rgb)
            state.image_embedding = state.predictor.get_image_embedding().cpu().numpy()
            embed_time = time.time() - t_embed_start
            perf_stats['embedding_times'].append(embed_time)

            # Prepare ONNX inputs
            onnx_coord_frame = state.predictor.transform.apply_coords(onnx_coord, frame_small.shape[:2]).astype(np.float32)
            ort_inputs = {
                "image_embeddings": state.image_embedding,
                "point_coords": onnx_coord_frame,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(frame_small.shape[:2], dtype=np.float32)
            }
            
            # Run inference
            t_inference_start = time.time()
            print("Running ONNX inference...")
            masks, scores, low_res_logits = state.ort_session.run(None, ort_inputs)
            
            # Apply threshold adjustment
            masks = masks > (state.predictor.model.mask_threshold + state.MASK_THRESHOLD_ADJUSTMENT)
            print(f"ONNX inference complete. Masks shape: {masks.shape}, Scores shape: {scores.shape}")

            inference_time = time.time() - t_inference_start
            perf_stats['inference_times'].append(inference_time)

            # Process inference results
            if masks is not None and len(masks) > 0:
                # Select top masks by score
                top_mask_indices = np.argsort(scores[0])[-2:]
                print(f"Selected top {len(top_mask_indices)} masks")
                
                # Resize masks to full frame size
                resized_masks = []
                frame_height, frame_width = frame_to_process.shape[:2]
                
                for mask_idx in top_mask_indices:
                    mask = masks[0][mask_idx]
                    print(f"Processing mask {mask_idx}: shape before resize={mask.shape}")
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    print(f"Mask {mask_idx}: shape after resize={mask_resized.shape}")
                    resized_masks.append(mask_resized)
                
                # Convert to numpy array for consistent handling
                resized_masks = np.array(resized_masks)
                
                # Update mask tracking
                update_mask_tracking(resized_masks)
                
                # Update shared state with thread safety
                with state.frame_lock:
                    state.current_masks = resized_masks
                    state.mask_scores = scores[0][top_mask_indices]
                    print(f"Updated current masks: {state.current_masks.shape}")
                    print(f"Current mask flags: {state.mask_flags}")
                    print(f"Stability counters: {state.mask_stability_counters}")
                    print(f"Mask change scores: {state.mask_change_scores}")
            else:
                print("No masks detected")
                with state.frame_lock:
                    state.current_masks = None
                    state.mask_scores = None

        except Exception as e:
            print(f"Error in inference loop: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)