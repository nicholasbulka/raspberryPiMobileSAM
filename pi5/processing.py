# processing.py
import cv2
import numpy as np
import time
import traceback
from picamera2 import Picamera2

from utils import (
    perf_stats, INPUT_SIZE, POINTS_PER_SIDE,
    generate_visualization, get_cpu_temp, log_performance_stats,
    update_mask_tracking, detect_motion
)
import shared_state as state
from effects import apply_effect

def capture_frames():
    """Continuously capture frames from the camera and update the current frame in shared state."""
    last_capture_time = time.time()
    
    try:
        state.picam2 = Picamera2()
        preview_config = state.picam2.create_preview_configuration(
            main={"size": INPUT_SIZE},
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
                
                # Sleep to cap at 60 FPS for capture
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
    """Process frames with current masks and apply visualizations."""
    print("Starting process frames thread")
    
    last_process_time = time.time()
    target_interval = 1/30  # 30 FPS target

    print("Waiting for first frame...")
    while state.current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        while True:
            try:
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                    frame_to_process = state.current_frame.copy()
                    previous_frame = state.previous_frame.copy() if state.previous_frame is not None else None
                    current_masks = state.current_masks
                    current_scores = state.mask_scores

                process_start_time = time.time()
                perf_stats['cpu_temps'].append(get_cpu_temp())
                
                current_time = time.time()
                time_since_last = current_time - last_process_time

                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue

                # Detect motion
                motion_mask, motion_intensity = detect_motion(frame_to_process, previous_frame)
                
                # Generate visualization with current masks and scores
                print("Generating visualization...")
                result = generate_visualization(frame_to_process, current_masks, current_scores)
                
                if motion_mask is not None:
                    print("\nDEBUG: Testing blending with artificial pattern")
                    
                    # Create a test pattern - a vertical stripe in the middle
                    test_mask = np.zeros((180, 320), dtype=np.float32)
                    # Make middle third of the image fully visible from original frame
                    test_mask[:, 107:214] = 1.0
                    
                    # Use our same reshaping operations
                    test_mask = test_mask[..., np.newaxis]
                    test_mask = np.repeat(test_mask, 3, axis=2)
                    
                    # Convert to float32 for calculations
                    result = result.astype(np.float32)
                    frame_to_process = frame_to_process.astype(np.float32)
                    
                    print("Test pattern statistics:")
                    print(f"Test mask shape: {test_mask.shape}")
                    print(f"Test mask unique values: {np.unique(test_mask)}")
                    print(f"Number of ones in test mask: {np.count_nonzero(test_mask)}")
                    
                    # Do the blending with our test pattern
                    result = (result * (1.0 - test_mask) + frame_to_process * test_mask).astype(np.uint8)
                    
                    print("\nBlending result statistics:")
                    print(f"Result min/max: {result.min()}, {result.max()}")
                    
                with state.frame_lock:
                    state.raw_processed_frame = result.copy()
                    state.processed_frame = result.copy()
                    state.motion_mask = motion_mask
                print("Visualization and motion processing complete")

                last_process_time = current_time
                perf_stats['total_process_times'].append(time.time() - process_start_time)
                
                # Log performance stats
                log_performance_stats()

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        traceback.print_exc()

def apply_effects_loop():
    """Apply visual effects to the processed frame using current masks."""
    print("Starting effects loop")
    while True:
        current_time = time.time()
        try:
            if state.current_effect is not None:
                print(f"\nEffect Debug:")
                print(f"Current effect: {state.current_effect}")
                print(f"Has raw frame: {state.raw_processed_frame is not None}")
                print(f"Has masks: {state.current_masks is not None}")
                
                with state.frame_lock:
                    if state.current_masks is not None:
                        print(f"Masks shape: {state.current_masks.shape}")
                        print(f"Mask flags: {state.mask_flags}")
                    
                    if state.raw_processed_frame is not None:
                        frame_to_process = state.raw_processed_frame.copy()
                        masks_for_effect = state.current_masks
                        
                        # Apply effect
                        result = apply_effect(frame_to_process, state.current_effect, state.effect_params, masks_for_effect)
                        
                        # Let motion break through the effect
                        if state.motion_mask is not None:
                            original_frame = state.current_frame.copy()
                            result = cv2.addWeighted(
                                result,
                                1 - state.motion_mask,
                                original_frame,
                                state.motion_mask,
                                0
                            )
                        
                        state.processed_frame = result
                    else:
                        print("No raw frame available")

            time.sleep(max(0, (1/60) - (time.time() - current_time)))  # Cap effects at 60 FPS

        except Exception as e:
            print(f"Effects error: {e}")
            traceback.print_exc()
            time.sleep(0.1)

def inference_loop(sam):
    """Run inference on frames and update mask tracking."""
    # Use smaller size for inference to maintain performance
    h, w = (180, 320)  # Reduced size for inference while keeping aspect ratio
    
    # Pre-calculate grid points for larger size
    points_per_side = POINTS_PER_SIDE  # Increase from 4 to get more detailed segments
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
            with state.frame_lock:
                if state.current_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_process = state.current_frame.copy()

            # Resize frame
            t_resize_start = time.time()
            frame_small = cv2.resize(frame_to_process, (w, h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            perf_stats['resize_times'].append(time.time() - t_resize_start)

            # Update embedding
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

            if masks is not None and len(masks) > 0:
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
                
                # Update tracking before we update the shared state
                update_mask_tracking(resized_masks)
                
                # Update shared state with thread safety
                with state.frame_lock:
                    state.current_masks = resized_masks
                    state.mask_scores = scores[0][top_mask_indices]  # Only keep scores for top masks
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