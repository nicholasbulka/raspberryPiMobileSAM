# processing.py
import cv2
import numpy as np
import time
import traceback
from picamera2 import Picamera2

from utils import (
    perf_stats, INPUT_SIZE,
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

                process_start_time = time.time()
                perf_stats['cpu_temps'].append(get_cpu_temp())
                
                current_time = time.time()
                time_since_last = current_time - last_process_time

                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue

                # Always generate visualization, using cached masks if available
                print("Generating visualization...")
                result = generate_visualization(frame_to_process, state.cached_masks, state.cached_scores)
                
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
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        traceback.print_exc()

def apply_effects_loop():
    print("Starting effects loop")
    while True:
        current_time = time.time()
        try:
            if state.current_effect is not None:
                print(f"\nEffect Debug:")
                print(f"Current effect: {state.current_effect}")
                print(f"Has raw frame: {state.raw_processed_frame is not None}")
                print(f"Has masks: {state.cached_masks is not None}")
                if state.cached_masks is not None:
                    print(f"Masks shape: {state.cached_masks.shape}")

                with state.frame_lock:
                    if state.raw_processed_frame is not None:
                        frame_to_process = state.raw_processed_frame.copy()
                        result = apply_effect(frame_to_process, state.current_effect, state.effect_params, state.cached_masks)
                        state.processed_frame = result
                    else:
                        print("No raw frame available")

            time.sleep(max(0, (1/60) - (time.time() - current_time)))  # Cap effects at 60 FPS

        except Exception as e:
            print(f"Effects error: {e}")
            traceback.print_exc()
            time.sleep(0.1)

def inference_loop(sam):
    print("Starting inference loop")
    
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

            t_embed_start = time.time()

            if first_frame:
                # First frame: do full embedding
                print("Computing initial full-frame embedding...")
                state.predictor.set_image(frame_rgb)
                state.image_embedding = state.predictor.get_image_embedding().cpu().numpy()
                first_frame = False

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
            
            # Run ONNX inference
            t_inference_start = time.time()

            print("Running ONNX inference...")
            masks, scores, low_res_logits = state.ort_session.run(None, ort_inputs)
            masks = masks > state.predictor.model.mask_threshold
            print(f"ONNX inference complete. Masks shape: {masks.shape}, Scores shape: {scores.shape}")

            inference_time = time.time() - t_inference_start
            perf_stats['inference_times'].append(inference_time)

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
                
                state.cached_masks = np.array(resized_masks)
                state.cached_scores = scores
                print(f"Updated cached masks: {state.cached_masks.shape}")
            else:
                print("No masks detected")

        except Exception as e:
            print(f"Error in inference loop: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)