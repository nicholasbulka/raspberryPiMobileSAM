import cv2 
import numpy as np
import time
import traceback
from picamera2 import Picamera2
from typing import List, Optional, Tuple

from utils import (
    perf_stats,
    get_cpu_temp,
    log_performance_stats,
    update_performance_stats
)
from motion import detect_motion
from mask import update_mask_tracking, create_physics_mask
from effects import apply_effect
import shared_state as state

DEBUG_MODE = False

def separate_objects(masks: np.ndarray,
                    scores: np.ndarray,
                    min_size: int = 100,
                    max_size: int = 45000,
                    min_confidence: float = 0.4,
                    iou_threshold: float = 0.1  # Lowered from 0.2 to allow more objects
                    ) -> List[np.ndarray]:
    """
    Separate individual objects using connected components analysis.
    """
    if masks is None or len(masks) == 0:
        print("No masks received in separate_objects")
        return []
    
    print(f"\nMask Debug:")
    print(f"Received masks shape: {masks.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Scores: {scores[0]}")
    
    object_masks = []
    total_pixels = masks[0][0].shape[0] * masks[0][0].shape[1]
    
    # Process each mask proposal
    for i in range(masks.shape[1]):
        if scores[0][i] < min_confidence:
            continue
            
        current_mask = masks[0][i]
        print(f"\nProcessing mask {i} with score {scores[0][i]}")
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            current_mask.astype(np.uint8),
            connectivity=4  # Changed from 8 to 4 for stricter connectivity
        )
        
        print(f"Found {num_labels-1} connected components")
        
        # Process each component
        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)
            size_percentage = (component_size / total_pixels) * 100
            
            print(f"  Component {label}: size={component_size} ({size_percentage:.1f}% of frame)")
            
            # Skip if component is too small or too large
            if component_size < min_size or component_size > max_size:
                print(f"    Skipping: size outside bounds ({min_size}-{max_size})")
                continue
                
            # Get component bounds
            y_indices, x_indices = np.where(component_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            
            # Skip if component touches edges
            if (y1 == 0 or y2 == component_mask.shape[0]-1 or
                x1 == 0 or x2 == component_mask.shape[1]-1):
                print("    Skipping: touches frame edge")
                continue
            
            # Check overlap with existing masks
            is_unique = True
            for existing_mask in object_masks:
                intersection = np.logical_and(component_mask, existing_mask)
                union = np.logical_or(component_mask, existing_mask)
                iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
                
                if iou > iou_threshold:
                    print(f"    Skipping: overlaps with existing mask (IoU={iou:.2f})")
                    is_unique = False
                    break
            
            if is_unique:
                print("    Accepting component")
                object_masks.append(component_mask)
    
    print(f"Found {len(object_masks)} valid masks")
    return object_masks

def capture_frames() -> None:
    """Continuously capture frames from the camera."""
    last_capture_time = time.time()
    
    try:
        # Initialize camera with optimal settings
        state.picam2 = Picamera2()
        preview_config = state.picam2.create_preview_configuration(
            main={"size": state.INPUT_SIZE},
            buffer_count=4
        )
        state.picam2.configure(preview_config)
        
        # Additional camera optimizations
        state.picam2.set_controls({
            "FrameDurationLimits": (16666, 100000),  # 10-60 FPS range
            "AwbEnable": True,
            "AeEnable": True,
            "NoiseReductionMode": 2  # Fast noise reduction
        })
        
        state.picam2.start()
        print("Camera started successfully")

        while True:
            try:
                # Capture frame
                frame = state.picam2.capture_array("main")
                current_time = time.time()
                
                # Track frame intervals
                frame_interval = current_time - last_capture_time
                update_performance_stats('frame_intervals', frame_interval)
                last_capture_time = current_time
                
                # Convert color space
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply camera flips if needed
                if state.camera_flipped_h:
                    frame = cv2.flip(frame, 1)
                if state.camera_flipped_v:
                    frame = cv2.flip(frame, 0)
                
                # Update shared state with thread safety
                with state.frame_lock:
                    if state.current_frame is not None:
                        state.previous_frame = state.current_frame.copy()
                    state.current_frame = frame.copy()
                
                # Rate limiting to maintain target FPS
                target_interval = 1/60  # Target 60 FPS
                elapsed = time.time() - current_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                
            except Exception as e:
                print(f"Frame capture error: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal camera error: {e}")
        traceback.print_exc()
    finally:
        if state.picam2 is not None:
            state.picam2.stop()
            print("Camera stopped")

def inference_loop(sam) -> None:
    """Run continuous mask inference with enhanced debugging."""
    print("Starting inference loop")
    
    # Use dimensions from shared state
    process_height, process_width = state.PROCESS_SIZE
    
    # Create fixed grid points
    points_per_side = state.POINTS_PER_SIDE
    x = np.linspace(0, process_width, points_per_side)
    y = np.linspace(0, process_height, points_per_side)
    xv, yv = np.meshgrid(x, y)
    fixed_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    
    # Set up ONNX inputs
    onnx_coord = np.concatenate([fixed_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([np.ones(len(fixed_points)), np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    print(f"Initialized inference loop with dimensions: {process_width}x{process_height}")
    print(f"Using {points_per_side}x{points_per_side} grid points")

    target_frame_time = 1/20  # Target 20 FPS for inference
    last_inference_time = time.time()
    frame_counter = 0

    while True:
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - last_inference_time < target_frame_time:
                time.sleep(target_frame_time - (current_time - last_inference_time))
                continue
            
            frame_counter += 1
            if frame_counter % 30 == 0:
                print(f"\n=== Frame {frame_counter} ===")
            
            # Get current frame
            with state.frame_lock:
                if state.current_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_process = state.current_frame.copy()
            
            # Process frame
            frame_small = cv2.resize(frame_to_process, (process_width, process_height))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Run SAM inference
            t_start = time.time()
            state.predictor.set_image(frame_rgb)
            state.image_embedding = state.predictor.get_image_embedding().cpu().numpy()
            
            # Transform coordinates
            onnx_coord_frame = state.predictor.transform.apply_coords(
                onnx_coord,
                frame_small.shape[:2]
            ).astype(np.float32)
            
            # Prepare ONNX inputs
            ort_inputs = {
                "image_embeddings": state.image_embedding,
                "point_coords": onnx_coord_frame,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(frame_small.shape[:2], dtype=np.float32)
            }
            
            # Run inference
            masks, scores, _ = state.ort_session.run(None, ort_inputs)
            update_performance_stats('inference_times', time.time() - t_start)
            
            # Apply threshold
            masks = masks > (state.predictor.model.mask_threshold + state.MASK_THRESHOLD_ADJUSTMENT)
            
            # Process masks
            if masks is not None and np.sum(scores[0] > 0.5) > 0:
                # Clear old inactive masks first
                with state.frame_lock:
                    if state.current_masks:
                        state.current_masks = [m for m in state.current_masks if m.is_active]
                
                object_masks = separate_objects(masks, scores)
                
                if object_masks:
                    # Create physics masks
                    frame_height, frame_width = frame_to_process.shape[:2]
                    physics_masks = []
                    
                    for object_mask in object_masks:
                        # Resize mask to full frame size
                        mask_resized = cv2.resize(
                            object_mask.astype(np.uint8),
                            (frame_width, frame_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        # Create initial pixel content by masking the original frame
                        pixel_content = np.zeros_like(frame_to_process)
                        pixel_content[mask_resized] = frame_to_process[mask_resized]
                        
                        # Debug print pixel content
                        print(f"Pixel content shape: {pixel_content.shape}")
                        print(f"Non-zero pixels: {np.count_nonzero(pixel_content)}")
                        print(f"Mask sum: {np.sum(mask_resized)}")
                        
                        # Create physics mask with pixel content
                        physics_mask = create_physics_mask(
                            mask_resized,
                            frame_to_process.shape[:2],
                            mass=1.0,
                            friction=0.35,
                            pixel_content=pixel_content
                        )
                        physics_masks.append(physics_mask)
                    
                    # Update shared state - merge with existing active masks
                    with state.frame_lock:
                        active_masks = [m for m in state.current_masks if m.is_active] if state.current_masks else []
                        state.current_masks = active_masks + physics_masks
                        state.mask_scores = [1.0] * len(state.current_masks)
                else:
                    with state.frame_lock:
                        # Keep only active masks
                        if state.current_masks:
                            state.current_masks = [m for m in state.current_masks if m.is_active]
                        state.mask_scores = [1.0] * len(state.current_masks) if state.current_masks else None
            
            last_inference_time = time.time()
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)

def process_frames() -> None:
    """
    Process captured frames with motion detection and prepare for visualization.
    """
    print("Starting frame processing")
    
    last_process_time = time.time()
    target_interval = 1/30  # Target 30 FPS
    
    # Wait for first frame
    while state.current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        while True:
            try:
                # Get latest frame safely
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                    frame_to_process = state.current_frame.copy()
                    previous_frame = state.previous_frame.copy() if state.previous_frame is not None else None

                # Track performance
                process_start = time.time()
                perf_stats['cpu_temps'].append(get_cpu_temp())
                
                # Rate limit processing
                current_time = time.time()
                time_since_last = current_time - last_process_time
                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue

                # Detect motion
                motion_contours, debug_frame, motion_intensity = detect_motion(frame_to_process, previous_frame)
                
                # Update mask tracking
                with state.frame_lock:
                    if state.current_masks:
                        update_mask_tracking(state.current_masks)
                
                # Update shared state
                with state.frame_lock:
                    state.raw_processed_frame = debug_frame if debug_frame is not None else frame_to_process.copy()
                    state.motion_contours = motion_contours
                    state.motion_intensity = motion_intensity

                # Update timing stats
                last_process_time = current_time
                perf_stats['total_process_times'].append(time.time() - process_start)
                
                # Log performance
                log_performance_stats()

            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal processing error: {str(e)}")
        traceback.print_exc()

def apply_effects_loop() -> None:
    """
    Apply visual effects to processed frames with support for physics-based effects.
    """
    print("Starting effects loop")
    target_interval = 1/60  # Target 60 FPS for effects

    while True:
        current_time = time.time()
        try:
            if state.current_effect is not None:
                # Get current state safely
                with state.frame_lock:
                    if state.raw_processed_frame is None:
                        continue
                    
                    frame_to_process = state.raw_processed_frame.copy()
                    masks_for_effect = state.current_masks
                    current_effect = state.current_effect
                    effect_params = state.effect_params.copy()
                
                # Apply selected effect
                result = apply_effect(
                    frame_to_process,
                    current_effect,
                    effect_params,
                    masks_for_effect
                )
                
                # Update shared state
                with state.frame_lock:
                    state.processed_frame = result

            # Rate limit effects processing
            elapsed = time.time() - current_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        except Exception as e:
            print(f"Effects error: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)