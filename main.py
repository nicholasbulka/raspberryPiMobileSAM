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
from flask import Flask, render_template_string, jsonify, request
import threading
import colorsys
print("Basic imports completed")

print("Importing effects module...")
from effects import apply_effect, get_javascript_code, EFFECTS_LIST
print("Effects module imported successfully")

# Initialize Flask app
print("Initializing Flask app...")
app = Flask(__name__)
print("Flask app initialized")

# Global variables for shared state
frame_lock = threading.Lock()
effect_lock = threading.Lock()
current_frame = None
raw_processed_frame = None  # Stores the raw segmentation result
processed_frame = None      # Stores the effect-processed frame
current_masks = None
current_effect = None
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
    return render_template_string('''
    <!DOCTYPE html>
    <html>
        <head>
            <title>MobileSAM Camera Stream</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background: #1a1a1a;
                    color: #fff;
                }
                .container {
                    position: relative;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: calc(100vh - 200px);
                }
                .stream {
                    position: absolute;
                    text-align: center;
                    background: #2a2a2a;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }
                .stream.raw {
                    display: none;
                    z-index: 1;
                }
                .stream.processed {
                    z-index: 2;
                }
                .canvas-container {
                    position: relative;
                    margin: 10px;
                }
                canvas {
                    max-width: 640px;
                    border: 2px solid #333;
                    background: #000;
                    border-radius: 5px;
                }
                .controls {
                    margin: 20px auto;
                    padding: 20px;
                    background: #333;
                    border-radius: 10px;
                    max-width: 800px;
                }
                .effect-btn, .toggle-btn {
                    padding: 10px 20px;
                    margin: 5px;
                    background: #444;
                    color: #fff;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-size: 14px;
                }
                .effect-btn:hover, .toggle-btn:hover {
                    background: #666;
                    transform: translateY(-2px);
                }
                .effect-btn.active, .toggle-btn.active {
                    background: #0066cc;
                    box-shadow: 0 0 10px rgba(0,102,204,0.5);
                }
                .slider-container {
                    margin: 15px 0;
                    background: #2a2a2a;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                .slider-container label {
                    display: inline-block;
                    width: 120px;
                    margin-right: 10px;
                }
                .slider {
                    width: 200px;
                    vertical-align: middle;
                }
                #global-controls {
                    margin-bottom: 20px;
                    padding: 10px;
                    border-bottom: 1px solid #444;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                #effect-buttons {
                    margin-bottom: 20px;
                    padding: 10px;
                    border-bottom: 1px solid #444;
                }
                #effect-params {
                    padding: 10px;
                }
                .title {
                    text-align: center;
                    color: #fff;
                    margin-bottom: 30px;
                    font-size: 24px;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }
                .show-raw-btn {
                    padding: 8px 16px;
                    background: #555;
                    border: none;
                    color: white;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .show-raw-btn:hover {
                    background: #666;
                }
                .show-raw-btn.active {
                    background: #0066cc;
                }
            </style>
            <script>
            ''' + get_javascript_code() + '''
            </script>
        </head>
        <body>
            <h1 class="title">MobileSAM Stream with Effects</h1>
            <div class="controls">
                <div id="global-controls">
                    <div>
                        <h2 style="margin-top: 0; display: inline-block; margin-right: 20px;">Global Controls</h2>
                        <button id="toggle-8bit" class="toggle-btn">8-bit Mode</button>
                    </div>
                    <button id="toggle-raw" class="show-raw-btn">Show Raw Feed</button>
                </div>
                <div id="effect-controls">
                    <h2 style="margin-top: 0;">Visual Effects</h2>
                    <div id="effect-buttons"></div>
                    <div id="effect-params"></div>
                </div>
            </div>
            <div class="container">
                <div class="stream raw">
                    <h2>Raw Feed</h2>
                    <div class="canvas-container">
                        <canvas id="rawCanvas"></canvas>
                    </div>
                </div>
                <div class="stream processed">
                    <h2>Processed Feed</h2>
                    <div class="canvas-container">
                        <canvas id="processedCanvas"></canvas>
                    </div>
                </div>
            </div>
            <script>
                document.getElementById('toggle-raw').addEventListener('click', function() {
                    const rawStream = document.querySelector('.stream.raw');
                    const btn = this;
                    if (rawStream.style.display === 'none' || !rawStream.style.display) {
                        rawStream.style.display = 'block';
                        btn.classList.add('active');
                        btn.textContent = 'Hide Raw Feed';
                    } else {
                        rawStream.style.display = 'none';
                        btn.classList.remove('active');
                        btn.textContent = 'Show Raw Feed';
                    }
                });
            </script>
        </body>
    </html>
    ''')

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

def capture_frames(camera_index=0):
    global current_frame
    print(f"Starting capture frames thread with camera index {camera_index}")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        print("Camera opened successfully")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            frame = cv2.flip(frame, -1)

            with frame_lock:
                current_frame = frame

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                elapsed = time.time() - start_time
                print(f"Capture FPS: {frame_count/elapsed:.2f}")

            time.sleep(0.033)

    except Exception as e:
        print(f"Error in capture thread: {e}")
    finally:
        print("Closing camera")
        cap.release()

def generate_visualization(frame, masks, scores):
    """Generate the base visualization without effects"""
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
            # Resize mask to match frame dimensions
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
    target_interval = 1.0

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
                time.sleep(0.1)
                continue

            try:
                with frame_lock:
                    if current_frame is None:
                        continue
                    frame_to_process = current_frame.copy()

                # Process with MobileSAM
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

                # Generate visualization
                result = generate_visualization(frame_to_process, masks, scores)

                with frame_lock:
                    raw_processed_frame = result.copy()
                    processed_frame = result.copy()

                last_process_time = current_time
                frames_processed += 1

                if frames_processed % 5 == 0:  # Log every 5 frames
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
                    if effects_applied % 30 == 0:  # Log every 30 effects
                        elapsed = time.time() - start_time
                        print(f"Effects FPS: {effects_applied/elapsed:.2f}")

                time.sleep(0.033)

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

    # Load MobileSAM model
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

        # Start threads
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

        # Start Flask server
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
