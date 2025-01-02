# shared_state.py
import threading
import numpy as np

# Global variables for shared state
frame_lock = threading.Lock()
effect_lock = threading.Lock()
current_frame = None
previous_frame = None  # Added for motion detection
motion_mask = None     # Added for motion detection
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

# Mask tracking variables
previous_masks = None  # Store previous iteration's masks
mask_change_scores = []  # Track how much each mask changed
mask_stability_counters = []  # Count how long masks have been stable
STABILITY_THRESHOLD = 0.75  # Reduced from 0.95 - how similar masks need to be to be considered stable
STABILITY_COUNT_THRESHOLD = 2  # Reduced from 5 - how many stable iterations before flagging
mask_flags = []  # Flags for each mask (e.g., 'stable', 'dynamic', etc.)

# Segmentation parameters
MASK_THRESHOLD_ADJUSTMENT = -0.4  # Default MobileSAM threshold (can be adjusted)