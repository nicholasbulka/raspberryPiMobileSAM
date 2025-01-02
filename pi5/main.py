# main.py

# Configure PyTorch for optimal performance on the device
import torch
print("Starting imports...")
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(2)  # Limit PyTorch threads

# Configure OpenCV threading
import cv2
cv2.setNumThreads(2)  # Limit OpenCV threads

# Standard library imports
import numpy as np
import time
import os
import io
import base64
from flask import Flask, render_template, jsonify, request
import threading
import onnxruntime
from collections import deque

# Local module imports
from processing import process_frames, inference_loop, capture_frames, apply_effects_loop
from rendering import render_loop  # New import for rendering functionality
from utils import (
    perf_stats, PERF_WINDOW_SIZE, LOG_INTERVAL, INPUT_SIZE, PROCESS_SIZE, TARGET_FPS,
    encode_frame_to_base64, cleanup
)
from effects import EFFECTS_LIST
import shared_state as state

print("Basic imports completed")

# Initialize Flask application
app = Flask(__name__)
print("Flask app initialized")

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html', effects=EFFECTS_LIST)

@app.route('/stream')
def stream():
    """Stream the processed video feed to the client."""
    with state.frame_lock:
        # Check if frames are available
        if state.current_frame is None or state.rendered_frame is None:
            return {"error": "No frames available"}, 404

        # Get the raw and rendered frames
        raw_b64 = encode_frame_to_base64(state.current_frame)
        rendered_b64 = encode_frame_to_base64(state.rendered_frame)

        # Create a copy of the current masks if they exist
        masks = state.current_masks.tolist() if state.current_masks is not None else None

    return {
        "raw": raw_b64,
        "processed": rendered_b64,  # Now using rendered_frame instead of processed_frame
        "masks": masks
    }

@app.route('/effect/<effect_name>', methods=['POST'])
def set_effect(effect_name):
    """Set the current visual effect."""
    print(f"\nSetting effect: {effect_name}")
    state.current_effect = effect_name if effect_name != 'none' else None
    print(f"Effect set to: {state.current_effect}")
    return jsonify({"status": "ok"})

@app.route('/effect_params', methods=['POST'])
def update_effect_params():
    """Update parameters for the current effect."""
    print(f"\nUpdating effect params: {request.json}")
    state.effect_params = request.json
    print(f"New params: {state.effect_params}")
    return jsonify({"status": "ok"})

@app.route('/flip_camera_h', methods=['POST'])
def flip_camera_h():
    """Toggle horizontal camera flip."""
    state.camera_flipped_h = not state.camera_flipped_h
    return jsonify({"status": "ok", "flipped": state.camera_flipped_h})

@app.route('/flip_camera_v', methods=['POST'])
def flip_camera_v():
    """Toggle vertical camera flip."""
    state.camera_flipped_v = not state.camera_flipped_v
    return jsonify({"status": "ok", "flipped": state.camera_flipped_v})

def main():
    """
    Main application entry point. Initializes models, starts processing threads,
    and runs the Flask server.
    """
    # Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)
    
    # Define model paths
    model_type = "vit_t"
    checkpoint = "weights/mobile_sam.pt"
    onnx_model_path = "weights/mobile_sam.onnx"

    # Check for required model files
    if not os.path.exists(checkpoint) or not os.path.exists(onnx_model_path):
        print("Error: Missing model files")
        return

    try:
        # Initialize models and ONNX session
        print("\n=== Starting Optimized MobileSAM Application ===")
        start_time = time.time()

        print(f"[{time.time() - start_time:.2f}s] Initializing models...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        print(f"[{time.time() - start_time:.2f}s] Model loaded")

        # Configure model for CPU execution
        print(f"[{time.time() - start_time:.2f}s] Configuring model...")
        sam.to(device='cpu')
        sam.eval()

        # Create optimized ONNX session
        session_options = onnxruntime.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 2

        # Initialize predictor and ONNX session
        state.predictor = SamPredictor(sam)
        state.ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )

        print(f"[{time.time() - start_time:.2f}s] ONNX session ready")

        # Start processing threads
        print(f"[{time.time() - start_time:.2f}s] Starting threads...")
        threads = [
            threading.Thread(target=capture_frames, daemon=True),
            threading.Thread(target=process_frames, daemon=True),
            threading.Thread(target=inference_loop, args=(sam,), daemon=True),
            threading.Thread(target=apply_effects_loop, daemon=True),
            threading.Thread(target=render_loop, daemon=True)  # New rendering thread
        ]

        print(f"[{time.time() - start_time:.2f}s] Threads starting...")

        # Start all threads
        for thread in threads:
            thread.start()

        # Start Flask server
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
        # Import sam model registry here to avoid circular imports
        from mobile_sam import sam_model_registry, SamPredictor
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        cleanup()