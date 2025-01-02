# main.py
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
import onnxruntime
from collections import deque

from processing import process_frames, inference_loop, capture_frames, apply_effects_loop
from utils import (
    perf_stats, PERF_WINDOW_SIZE, LOG_INTERVAL, INPUT_SIZE, PROCESS_SIZE, TARGET_FPS,
    encode_frame_to_base64, cleanup
)
from effects import EFFECTS_LIST
import shared_state as state

print("Basic imports completed")

app = Flask(__name__)
print("Flask app initialized")

@app.route('/')
def index():
    return render_template('index.html', effects=EFFECTS_LIST)

@app.route('/stream')
def stream():
    with state.frame_lock:
        if state.current_frame is None or state.processed_frame is None:
            return {"error": "No frames available"}, 404

        raw_b64 = encode_frame_to_base64(state.current_frame)
        processed_b64 = encode_frame_to_base64(state.processed_frame)

        # Create a copy of the current masks if they exist
        masks = state.current_masks.tolist() if state.current_masks is not None else None

    return {
        "raw": raw_b64,
        "processed": processed_b64,
        "masks": masks
    }

@app.route('/effect/<effect_name>', methods=['POST'])
def set_effect(effect_name):
    print(f"\nSetting effect: {effect_name}")
    state.current_effect = effect_name if effect_name != 'none' else None
    print(f"Effect set to: {state.current_effect}")
    return jsonify({"status": "ok"})

@app.route('/effect_params', methods=['POST'])
def update_effect_params():
    print(f"\nUpdating effect params: {request.json}")
    state.effect_params = request.json
    print(f"New params: {state.effect_params}")
    return jsonify({"status": "ok"})

@app.route('/flip_camera_h', methods=['POST'])
def flip_camera_h():
    state.camera_flipped_h = not state.camera_flipped_h
    return jsonify({"status": "ok", "flipped": state.camera_flipped_h})

@app.route('/flip_camera_v', methods=['POST'])
def flip_camera_v():
    state.camera_flipped_v = not state.camera_flipped_v
    return jsonify({"status": "ok", "flipped": state.camera_flipped_v})

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

        state.predictor = SamPredictor(sam)
        state.ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )

        # Start threads
        threads = [
            threading.Thread(target=capture_frames, daemon=True),
            threading.Thread(target=process_frames, daemon=True),
            threading.Thread(target=inference_loop, args=(sam,), daemon=True),
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