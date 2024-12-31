# processing.py
import cv2
import numpy as np
import time
import traceback
from picamera2 import Picamera2

from utils import (
    perf_stats, INPUT_SIZE, get_change_bbox, update_embedding_region,
    generate_visualization, get_cpu_temp, log_performance_stats
)
import shared_state as state
from effects import apply_effect

def capture_frames():
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
                    state.current_frame = frame.copy()
                
                time.sleep(1/60)  # Cap at 60 FPS for capture
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal capture error: {e}")
        traceback.print_exc()
    finally:
        if state.picam2 is not None:
            state.picam2.stop()

def process_frames(sam):
    print("Starting process frames thread")
    
    last_process_time = time.time()
    target_interval = 0.125  # 8 FPS target
    embedding_size = (32, 32)  # Initial embedding size
    prev_frame = None
    first_frame = True  # Track if this is our first frame

    # Pre-calculate grid points for 96x54 processing size
    h, w = (54, 96)
    points_per_side = 4
    x = np.linspace(0, w, points_per_side)
    y = np.linspace(0, h, points_per_side)
    xv, yv = np.meshgrid(x, y)
    fixed_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    
    # Pre-allocate arrays
    onnx_coord = np.concatenate([fixed_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([np.ones(len(fixed_points)), np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    print("Waiting for first frame...")
    while state.current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        with state.frame_lock:
            state.processed_frame = state.current_frame.copy()
            state.raw_processed_frame = state.current_frame.copy()

        consecutive_failures = 0
        while True:
            try:
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                    frame_to_process = state.current_frame.copy()

                # Even if mask generation fails, we should still show the video
                with state.frame_lock:
                    state.raw_processed_frame = frame_to_process.copy()
                    state.processed_frame = frame_to_process.copy()

                process_start_time = time.time()
                perf_stats['cpu_temps'].append(get_cpu_temp())
                
                current_time = time.time()
                time_since_last = current_time - last_process_time

                if time_since_last < target_interval * 0.8:
                    time.sleep(0.001)
                    continue
                elif time_since_last > target_interval * 2:
                    print("Skipping frame to catch up")
                    last_process_time = current_time
                    continue

                # Resize timing
                t_resize_start = time.time()
                frame_small = cv2.resize(frame_to_process, (w, h), interpolation=cv2.INTER_LINEAR)
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                perf_stats['resize_times'].append(time.time() - t_resize_start)

                # Embedding timing
                t_embed_start = time.time()
                
                if first_frame:
                    # First frame: do full embedding
                    print("Computing initial full-frame embedding...")
                    state.predictor.set_image(frame_rgb)
                    state.image_embedding = state.predictor.get_image_embedding().cpu().numpy()
                    first_frame = False
                else:
                    # Find region of change
                    bbox = get_change_bbox(frame_small, prev_frame)
                    
                    if bbox is not None:
                        x, y, w, h = bbox
                        print(f"Change detected: {w}x{h} region at ({x},{y})")
                        
                        # Extract region of interest
                        roi = frame_rgb[y:y+h, x:x+w]
                        
                        # Compute embedding for ROI
                        state.predictor.set_image(roi)
                        roi_embedding = state.predictor.get_image_embedding().cpu().numpy()
                        
                        # Update the embedding with the new region
                        state.image_embedding = update_embedding_region(
                            state.image_embedding, 
                            roi_embedding,
                            x, y, w, h,
                            frame_small.shape[:2]
                        )
                    
                embed_time = time.time() - t_embed_start
                perf_stats['embedding_times'].append(embed_time)
                
                # Store frame for next comparison
                prev_frame = frame_small.copy()

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

                # Inference timing
                t_infer_start = time.time()
                print("Running ONNX inference...")
                masks, scores, low_res_logits = state.ort_session.run(None, ort_inputs)
                masks = masks > state.predictor.model.mask_threshold
                print(f"ONNX inference complete. Masks shape: {masks.shape}, Scores shape: {scores.shape}")
                perf_stats['inference_times'].append(time.time() - t_infer_start)

                if masks is not None and len(masks) > 0:
                    top_mask_indices = np.argsort(scores[0])[-2:]
                    print(f"Selected top {len(top_mask_indices)} masks")
                    # Resize masks to full frame size before storing
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
                    
                    state.current_masks = np.array(resized_masks)
                    print(f"Final masks shape: {state.current_masks.shape}")
                    current_scores = scores[0][top_mask_indices]
                    consecutive_failures = 0
                else:
                    print("No masks detected")
                    state.current_masks = None
                    current_scores = None

                if state.current_masks is not None:
                    print("Generating visualization...")
                    result = generate_visualization(frame_to_process, state.current_masks, current_scores)
                    with state.frame_lock:
                        state.raw_processed_frame = result.copy()
                        state.processed_frame = result.copy()
                    print("Visualization applied to frame")

                last_process_time = current_time
                perf_stats['total_process_times'].append(time.time() - process_start_time)
                
                # Log performance stats
                log_performance_stats()

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                consecutive_failures += 1
                if consecutive_failures > 5:
                    print("Too many consecutive failures, sleeping longer")
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        traceback.print_exc()

def apply_effects_loop():
    print("Starting effects loop")
    while True:
        try:
            if state.current_effect is not None:
                print(f"\nEffect Debug:")
                print(f"Current effect: {state.current_effect}")
                print(f"Has raw frame: {state.raw_processed_frame is not None}")
                print(f"Has masks: {state.current_masks is not None}")
                if state.current_masks is not None:
                    print(f"Masks shape: {state.current_masks.shape}")

                with state.frame_lock:
                    if state.raw_processed_frame is not None:
                        frame_to_process = state.raw_processed_frame.copy()
                        result = apply_effect(frame_to_process, state.current_effect, state.effect_params, state.current_masks)
                        state.processed_frame = result
                    else:
                        print("No raw frame available")

            time.sleep(1/60)  # Cap effects at 60 FPS

        except Exception as e:
            print(f"Effects error: {e}")
            traceback.print_exc()
            time.sleep(0.1)
