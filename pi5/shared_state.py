# shared_state.py
# This module manages shared state between different threads in the application.
# It provides thread-safe access to frames, masks, and processing parameters.

import threading
import numpy as np

# Thread synchronization locks
frame_lock = threading.Lock()        # Protects frame-related state
effect_lock = threading.Lock()       # Protects effect-related state

# Frame pipeline state
# These variables represent different stages of frame processing
current_frame = None        # Raw frame from camera
previous_frame = None       # Previous frame for motion detection
motion_mask = None         # Current motion detection mask
raw_processed_frame = None  # Frame after initial processing
processed_frame = None     # Frame after effects
rendered_frame = None      # Final frame after rendering (new!)

# Mask processing state
current_masks = None       # Current segmentation masks
mask_scores = None        # Confidence scores for current masks
previous_masks = None     # Previous frame's masks for tracking
mask_change_scores = []   # Track mask changes between frames
mask_stability_counters = []  # Count frames of mask stability
mask_flags = []          # Flags for each mask (e.g., 'stable', 'dynamic')

# Camera configuration
camera_flipped_h = False   # Horizontal flip state
camera_flipped_v = False   # Vertical flip state

# Effect state
current_effect = None      # Currently active visual effect
effect_params = {}         # Parameters for current effect

# Model instances and embeddings
predictor = None           # SAM predictor instance
ort_session = None        # ONNX runtime session
image_embedding = None    # Current frame's embedding
picam2 = None            # Global picamera2 instance

# Segmentation parameters and thresholds
STABILITY_THRESHOLD = 0.75        # How similar masks need to be to be considered stable
STABILITY_COUNT_THRESHOLD = 2     # How many stable iterations before flagging
MASK_THRESHOLD_ADJUSTMENT = -0.4  # Adjustment to MobileSAM threshold