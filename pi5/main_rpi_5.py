# Initialize Flask app and configuration
import torch
print("Starting imports...")
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(4)

import cv2
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
print("Basic imports completed")

print("Importing effects module...")
from effects import apply_effect, EFFECTS_LIST
print("Effects module imported successfully")

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

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000
        return temp
    except:
        return 0.0

def encode_frame_to_base64(frame):
    if frame is None:
        return None
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    print("Rendering index page...")
    return render_template('index.html', effects=EFFECTS_LIST)

@app.route('/stream')
def stream():
    print("Stream endpoint accessed")
    with frame_lock:
        if current_frame is None or processed_frame is None:
            print("No frames available")
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
    print(f"Setting effect: {effect_name}")
    current_effect = effect_name if effect_name != 'none' else None
    return jsonify({"status": "ok"})

@app.route('/effect_params', methods=['POST'])
def update_effect_params():
    global effect_params
    print("Updating effect parameters")
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


def capture_frames():
    global current_frame, camera_flipped_h, camera_flipped_v
    print("Starting capture frames thread with Picamera2")
    
    try:
        # Initialize the camera
        picam2 = Picamera2()
        
        # Configure the camera
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 360)},
            buffer_count=4
        )
        picam2.configure(preview_config)
        picam2.start()
        print("Camera started successfully")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Capture frame
                frame = picam2.capture_array("main")
                
                # Convert from RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply flips based on toggle states
                if camera_flipped_h:
                    frame = cv2.flip(frame, 1)  # 1 for horizontal flip
                if camera_flipped_v:
                    frame = cv2.flip(frame, 0)  # 0 for vertical flip
                
                with frame_lock:
                    current_frame = frame.copy()
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count/elapsed
                    print(f"Capture FPS: {fps:.2f}")
                    
                    if frame_count >= 120:
                        frame_count = 0
                        start_time = time.time()
                
                time.sleep(0.016)
                
            except Exception as e:
                print(f"Error during frame capture: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal error in capture thread: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing camera")
        try:
            picam2.stop()
            print("Camera stopped successfully")
        except:
            print("Error stopping camera")

def generate_visualization(frame, masks, scores):
    print("Generating visualization...")
    result = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    if masks is not None and len(masks) > 0:
        colors = generate_vibrant_colors(len(masks))

        if scores is not None:
            mask_score_pairs = list(zip(masks, scores, colors))
            mask_score_pairs.sort(key=lambda x: x[1], reverse=True)
            masks, scores, colors = zip(*mask_score_pairs)

        for i, (mask, color) in enumerate(zip(masks, colors)):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (frame_width, frame_height), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            overlay = np.zeros_like(frame)
            overlay[mask_resized] = color

            base_alpha = 0.25 + (0.15 * (i / len(masks)))
            result = cv2.addWeighted(result, 1, overlay, base_alpha, 0)

            mask_uint8 = mask_resized.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            thickness = 2 if i % 2 == 0 else 1
            contour_color = np.clip(color + np.array([128, 128, 128]), 0, 255)
            cv2.drawContours(result, contours, -1, contour_color.tolist(), thickness)

    print("Visualization complete")
    return result

def process_frames(predictor):
    global processed_frame, raw_processed_frame, current_frame, current_masks
    print("Starting process frames thread")
    
    frames_processed = 0
    start_time = time.time()
    last_process_time = 0
    target_interval = 0.033  # Reduced to ~30 FPS for faster updates

    print("Waiting for first frame...")
    while current_frame is None:
        time.sleep(0.1)
    print("First frame received")

    try:
        with frame_lock:
            processed_frame = current_frame.copy()
            raw_processed_frame = current_frame.copy()
        print("Initial frames copied")

        while True:
            current_time = time.time()

            if current_time - last_process_time < target_interval:
                continue

            try:
                with frame_lock:
                    if current_frame is None:
                        continue
                    frame_to_process = current_frame.copy()

                frame_small = cv2.resize(frame_to_process, (128, 72))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

                predictor.set_image(frame_rgb)

                h, w = frame_small.shape[:2]
                points_per_side = 2
                x = np.linspace(0, w, points_per_side)
                y = np.linspace(0, h, points_per_side)
                xv, yv = np.meshgrid(x, y)
                points = np.stack([xv.flatten(), yv.flatten()], axis=1)

                print("Running prediction...")
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=np.ones(len(points)),
                        multimask_output=True
                    )
                print("Prediction completed")

                if masks is not None and len(masks) > 0:
                    current_masks = np.array(masks)
                    print(f"Generated {len(masks)} masks")
                else:
                    current_masks = None
                    print("No masks generated")

                result = generate_visualization(frame_to_process, masks, scores)

                with frame_lock:
                    raw_processed_frame = result.copy()
                    processed_frame = result.copy()

                last_process_time = current_time
                frames_processed += 1

                if frames_processed % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processing FPS: {frames_processed/elapsed:.2f}")

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in process thread: {str(e)}")
        import traceback
        traceback.print_exc()

def apply_effects_loop():
    global processed_frame, raw_processed_frame
    print("Starting effects loop thread")
    
    effects_applied = 0
    start_time = time.time()

    try:
        while True:
            try:
                if current_effect is not None and raw_processed_frame is not None:
                    print(f"Applying effect: {current_effect}")
                    with frame_lock:
                        frame_to_process = raw_processed_frame.copy()

                    result = apply_effect(frame_to_process, current_effect, effect_params, current_masks)

                    with frame_lock:
                        processed_frame = result

                    effects_applied += 1
                    if effects_applied % 30 == 0:
                        elapsed = time.time() - start_time
                        print(f"Effects FPS: {effects_applied/elapsed:.2f}")

                time.sleep(0.016)  # Reduced to ~60 FPS max

            except Exception as e:
                print(f"Error in effects loop: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error in effects thread: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_vibrant_colors(n):
    colors = []
    golden_ratio = 0.618033988749895
    saturation_range = (0.7, 1.0)
    value_range = (0.8, 1.0)

    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        saturation = np.random.uniform(*saturation_range)
        value = np.random.uniform(*value_range)
        rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
        variation = np.random.uniform(-20, 20, 3)
        rgb = np.clip(rgb + variation, 0, 255)
        colors.append(rgb.astype(np.uint8))

    np.random.shuffle(colors)
    return colors

def main():
    print("\n=== Starting MobileSAM Application ===")
    print("Current working directory:", os.getcwd())

    os.makedirs("weights", exist_ok=True)
    print("Weights directory checked")

    print("Loading MobileSAM model...")
    model_type = "vit_t"
    checkpoint = "weights/mobile_sam.pt"

    if not os.path.exists(checkpoint):
        print(f"Error: Model file not found at {checkpoint}")
        print("Please download MobileSAM weights using:")
        print("wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P weights/")
        return

    try:
        print("Initializing model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = sam_model_registry[model_type](checkpoint=checkpoint)
        print("Model created")

        model.to(device=device)
        print("Model moved to device")

        model.eval()
        print("Model set to eval mode")

        predictor = SamPredictor(model)
        print("Predictor created")

        print("\nStarting threads...")

        print("Starting capture thread...")
        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        print("Capture thread started")

        print("Starting process thread...")
        process_thread = threading.Thread(target=process_frames, args=(predictor,))
        process_thread.daemon = True
        process_thread.start()
        print("Process thread started")

        print("Starting effects thread...")
        effects_thread = threading.Thread(target=apply_effects_loop)
        effects_thread.daemon = True
        effects_thread.start()
        print("Effects thread started")

        print("\nStarting Flask server...")
        print("Server will be available at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)

    except Exception as e:
        print(f"\nFatal error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nUnhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
