import torch
print("Starting imports...")
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(2)  # Limit PyTorch threads

import cv2
cv2.setNumThreads(2)  # Limit OpenCV threads
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import time
import os
import io
import base64
from flask import Flask, render_template, jsonify, request
import threading
import colorsys
from picamera2 import Picamera2
import onnxruntime
from collections import deque
import statistics
print("Basic imports completed")

print("Importing effects module...")
from effects import apply_effect, EFFECTS_LIST
print("Effects module imported successfully")

# Constants for optimization
INPUT_SIZE = (320, 180)  # Camera capture size
PROCESS_SIZE = (64, 36)  # SAM processing size
MAX_MASKS = 3  # Maximum number of masks to process
VISUALIZATION_SCALE = 0.5  # Scale factor for mask visualization
POINTS_PER_SIDE = 2  # Number of points for SAM
TARGET_FPS = 10  # Target FPS for processing

# Performance monitoring constants
PERF_WINDOW_SIZE = 100  # Number of frames to keep stats for
LOG_INTERVAL = 5.0  # Seconds between logging performance stats

app = Flask(__name__)
print("Flask app initialized")

# Global variables for shared state
frame_lock = threading.Lock()
effect_lock = threading.Lock()
current_frame = None
raw_processed_frame = None
processed_frame = None
current_masks = None
current_effect = None
camera_flipped_h = False
camera_flipped_v = False
effect_params = {}
predictor = None
ort_session = None
image_embedding = None
picam2 = None  # Global picamera2 instance

# Performance monitoring variables
perf_stats = {
    'resize_times': deque(maxlen=PERF_WINDOW_SIZE),
    'embedding_times': deque(maxlen=PERF_WINDOW_SIZE),
    'inference_times': deque(maxlen=PERF_WINDOW_SIZE),
    'visualization_times': deque(maxlen=PERF_WINDOW_SIZE),
    'total_process_times': deque(maxlen=PERF_WINDOW_SIZE),
    'frame_intervals': deque(maxlen=PERF_WINDOW_SIZE),
    'cpu_temps': deque(maxlen=PERF_WINDOW_SIZE)
}

last_log_time = time.time()

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000
        return temp
    except:
        return 0.0

def log_performance_stats():
    global last_log_time
    current_time = time.time()
    
    if current_time - last_log_time < LOG_INTERVAL:
        return
    
    stats = {}
    for key, values in perf_stats.items():
        if values:
            stats[key] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values)
            }
    
    print("\n=== Performance Statistics ===")
    print(f"CPU Temperature: {stats['cpu_temps']['mean']:.1f}°C")
    print(f"Frame Intervals: {stats['frame_intervals']['mean']*1000:.1f}ms (FPS: {1/stats['frame_intervals']['mean']:.1f})")
    print("\nProcessing Times (ms):")
    print(f"Resize:        {stats['resize_times']['mean']*1000:.1f}")
    print(f"Embedding:     {stats['embedding_times']['mean']*1000:.1f}")
    print(f"Inference:     {stats['inference_times']['mean']*1000:.1f}")
    print(f"Visualization: {stats['visualization_times']['mean']*1000:.1f}")
    print(f"Total:         {stats['total_process_times']['mean']*1000:.1f}")
    print("===========================\n")
    
    last_log_time = current_time

def encode_frame_to_base64(frame):
    if frame is None:
        return None
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html', effects=EFFECTS_LIST)

@app.route('/stream')
def stream():
    with frame_lock:
        if current_frame is None or processed_frame is None:
            return {"error": "No frames available"}, 404

        raw_b64 = encode_frame_to_base64(current_frame)
        processed_b64 = encode_frame_to_base64(processed_frame)

    return {
        "raw": raw_b64,
        "processed": processed_b64,
        "masks": current_masks.tolist() if current_masks is not None else None
    }

@app.route('/effect/<effect_name>', methods=['POST'])
def set_effect(effect_name):
    global current_effect
    current_effect = effect_name if effect_name != 'none' else None
    return jsonify({"status": "ok"})

@app.route('/effect_params', methods=['POST'])
def update_effect_params():
    global effect_params
    effect_params = request.json
    return jsonify({"status": "ok"})

@app.route('/flip_camera_h', methods=['POST'])
def flip_camera_h():
    global camera_flipped_h
    camera_flipped_h = not camera_flipped_h
    return jsonify({"status": "ok", "flipped": camera_flipped_h})

@app.route('/flip_camera_v', methods=['POST'])
def flip_camera_v():
    global camera_flipped_v
    camera_flipped_v = not camera_flipped_v
    return jsonify({"status": "ok", "flipped": camera_flipped_v})

def generate_vibrant_colors(n):
    colors = []
    golden_ratio = 0.618033988749895
    hues = [(i * golden_ratio) % 1.0 for i in range(n)]
    
    for hue in hues:
        rgb = np.array(colorsys.hsv_to_rgb(hue, 0.9, 0.9)) * 255
        colors.append(rgb.astype(np.uint8))
    
    return colors

def fast_visualization(frame, masks, scores):
    t_start = time.time()
    result = frame.copy()
    if masks is None or len(masks) == 0:
        perf_stats['visualization_times'].append(time.time() - t_start)
        return result

    frame_height, frame_width = frame.shape[:2]
    target_height = int(frame_height * VISUALIZATION_SCALE)
    target_width = int(frame_width * VISUALIZATION_SCALE)
    
    masks = masks[:MAX_MASKS]
    if scores is not None:
        scores = scores[:MAX_MASKS]
    
    colors = generate_vibrant_colors(len(masks))
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_small = cv2.resize(
            mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )
        mask_resized = cv2.resize(
            mask_small,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        mask_indices = np.where(mask_resized)
        if len(mask_indices[0]) > 0:
            result[mask_indices] = result[mask_indices] * 0.7 + color * 0.3

        if i == 0:
            mask_uint8 = mask_resized.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color.tolist(), 1)

    perf_stats['visualization_times'].append(time.time() - t_start)
    return result

def capture_frames():
    global current_frame, camera_flipped_h, camera_flipped_v, picam2
    last_capture_time = time.time()
    
    try:
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(
            main={"size": INPUT_SIZE},
            buffer_count=4
        )
        picam2.configure(preview_config)
        picam2.start()

        while True:
            try:
                frame = picam2.capture_array("main")
                current_time = time.time()
                perf_stats['frame_intervals'].append(current_time - last_capture_time)
                last_capture_time = current_time
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if camera_flipped_h:
                    frame = cv2.flip(frame, 1)
                if camera_flipped_v:
                    frame = cv2.flip(frame, 0)
                
                with frame_lock:
                    current_frame = frame.copy()
                
                time.sleep(1/60)  # Cap at 60 FPS for capture
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal capture error: {e}")
    finally:
        if picam2 is not None:
            picam2.stop()

def get_change_bbox(current_frame, previous_frame, min_area=100, max_size=128):
    """
    Detect the bounding box of the largest changed region between frames.
    Returns: (x, y, w, h) or None if no significant change
    """
    if previous_frame is None:
        return None
        
    # Compute difference between frames
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Add some blur to reduce noise
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    
    # Threshold and find contours
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return None
        
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding to the bbox (20% on each side)
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(current_frame.shape[1] - x, w + 2 * pad_x)
    h = min(current_frame.shape[0] - y, h + 2 * pad_y)
    
    # Limit size while maintaining aspect ratio
    if w > max_size or h > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        return x, y, new_w, new_h
    
    return x, y, w, h

def update_embedding_region(full_embedding, roi_embedding, x, y, w, h, frame_size):
    """
    Carefully update a region of the full embedding with the ROI embedding
    """
    # Get embedding spatial dimensions
    _, channels, embed_h, embed_w = full_embedding.shape
    
    # Calculate corresponding position in embedding space
    embed_x = int(x * embed_w / frame_size[1])
    embed_y = int(y * embed_h / frame_size[0])
    embed_roi_w = int(w * embed_w / frame_size[1])
    embed_roi_h = int(h * embed_h / frame_size[0])
    
    # Ensure we don't go out of bounds
    embed_roi_w = min(embed_roi_w, embed_w - embed_x)
    embed_roi_h = min(embed_roi_h, embed_h - embed_y)
    
    # Resize ROI embedding to match the target size
    roi_resized = []
    for c in range(channels):
        channel = cv2.resize(
            roi_embedding[0, c],
            (embed_roi_w, embed_roi_h),
            interpolation=cv2.INTER_LINEAR
        )
        roi_resized.append(channel)
    roi_resized = np.stack(roi_resized)
    
    # Create a blending mask for smooth transition
    blend_mask = np.ones((embed_roi_h, embed_roi_w), dtype=np.float32)
    blend_border = 2
    if blend_border > 0:
        blend_mask[:blend_border, :] = np.linspace(0, 1, blend_border)[:, np.newaxis]
        blend_mask[-blend_border:, :] = np.linspace(1, 0, blend_border)[:, np.newaxis]
        blend_mask[:, :blend_border] *= np.linspace(0, 1, blend_border)[np.newaxis, :]
        blend_mask[:, -blend_border:] *= np.linspace(1, 0, blend_border)[np.newaxis, :]
    
    # Apply blended update
    for c in range(channels):
        current = full_embedding[0, c, embed_y:embed_y+embed_roi_h, embed_x:embed_x+embed_roi_w]
        updated = roi_resized[c]
        full_embedding[0, c, embed_y:embed_y+embed_roi_h, embed_x:embed_x+embed_roi_w] = \
            current * (1 - blend_mask) + updated * blend_mask
    
    return full_embedding

def process_frames(sam):
    global processed_frame, raw_processed_frame, current_frame, current_masks, predictor, image_embedding
    print("Starting process frames thread")
    
    last_process_time = time.time()
    target_interval = 0.125  # 8 FPS target
    embedding_size = (32, 32)  # Initial embedding size
    prev_frame = None
    first_frame = True  # Track if this is our first frame

    # Pre-calculate grid points for 96x54 processing size
    h, w = (54, 96)
    points_per_side = 4

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
    while current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        with frame_lock:
            processed_frame = current_frame.copy()
            raw_processed_frame = current_frame.copy()

        consecutive_failures = 0
        while True:
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

            try:
                with frame_lock:
                    if current_frame is None:
                        continue
                    frame_to_process = current_frame.copy()

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
                    predictor.set_image(frame_rgb)
                    image_embedding = predictor.get_image_embedding().cpu().numpy()
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
                        predictor.set_image(roi)
                        roi_embedding = predictor.get_image_embedding().cpu().numpy()
                        
                        # Update the embedding with the new region
                        image_embedding = update_embedding_region(
                            image_embedding, 
                            roi_embedding,
                            x, y, w, h,
                            frame_small.shape[:2]
                        )
                    
                embed_time = time.time() - t_embed_start
                perf_stats['embedding_times'].append(embed_time)
                
                # Store frame for next comparison
                prev_frame = frame_small.copy()

                # Prepare ONNX inputs
                onnx_coord_frame = predictor.transform.apply_coords(onnx_coord, frame_small.shape[:2]).astype(np.float32)
                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord_frame,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(frame_small.shape[:2], dtype=np.float32)
                }

                # Inference timing
                t_infer_start = time.time()
                masks, scores, low_res_logits = ort_session.run(None, ort_inputs)
                masks = masks > predictor.model.mask_threshold
                perf_stats['inference_times'].append(time.time() - t_infer_start)

                if masks is not None and len(masks) > 0:
                    top_mask_indices = np.argsort(scores[0])[-2:]
                    current_masks = np.array(masks[0][top_mask_indices])
                    current_scores = scores[0][top_mask_indices]
                    consecutive_failures = 0
                else:
                    current_masks = None
                    current_scores = None

                if current_masks is not None:
                    result = fast_visualization(frame_to_process, current_masks, current_scores)

                    with frame_lock:
                        raw_processed_frame = result.copy()
                        processed_frame = result.copy()

                last_process_time = current_time
                perf_stats['total_process_times'].append(time.time() - process_start_time)
                
                # Log performance stats
                log_performance_stats()

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures > 5:
                    print("Too many consecutive failures, sleeping longer")
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        import traceback
        traceback.print_exc()

def apply_effects_loop():
    global processed_frame, raw_processed_frame
    
    while True:
        try:
            if current_effect is not None and raw_processed_frame is not None:
                with frame_lock:
                    frame_to_process = raw_processed_frame.copy()

                result = apply_effect(frame_to_process, current_effect, effect_params, current_masks)

                with frame_lock:
                    processed_frame = result

            time.sleep(1/60)  # Cap effects at 60 FPS

        except Exception as e:
            print(f"Effects error: {e}")
            time.sleep(0.1)

def cleanup():
    global picam2
    if picam2 is not None:
        try:
            picam2.stop()
        except:
            pass

def main():
    print("\n=== Starting Optimized MobileSAM Application ===")
    
    os.makedirs("weights", exist_ok=True)
    
    model_type = "vit_t"
    checkpoint = "weights/mobile_sam.pt"
    onnx_model_path = "weights/mobile_sam.onnx"

    if not os.path.exists(checkpoint) or not os.path.exists(onnx_model_path):
        print("Error: Missing model files")
        return

    try:
        # Initialize models
        print("Initializing models on CPU...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cpu')
        sam.eval()

        # Create optimized ONNX session
        session_options = onnxruntime.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 2

        global predictor, ort_session
        predictor = SamPredictor(sam)
        ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )

        # Start threads
        threads = [
            threading.Thread(target=capture_frames, daemon=True),
            threading.Thread(target=process_frames, args=(sam,), daemon=True),
            threading.Thread(target=apply_effects_loop, daemon=True)
        ]
        
        for thread in threads:
            thread.start()

        print(f"\nStarting Flask server (Input: {INPUT_SIZE}, Process: {PROCESS_SIZE}, Target FPS: {TARGET_FPS})")
        app.run(host='0.0.0.0', port=5000, threaded=True)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
