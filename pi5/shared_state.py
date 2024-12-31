# shared_state.py
import threading

# Global variables for shared state
frame_lock = threading.Lock()
effect_lock = threading.Lock()
current_frame = None
raw_processed_frame = None
processed_frame = None
current_masks = None
cached_masks = None
cached_scores = None
current_effect = None
camera_flipped_h = False
camera_flipped_v = False
effect_params = {}
predictor = None
ort_session = None
image_embedding = None
picam2 = None  # Global picamera2 instance