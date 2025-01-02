# utils.py
import cv2
import numpy as np
import colorsys
import base64
import time
import statistics
from collections import deque
import traceback
import shared_state as state

# Constants for optimization
INPUT_SIZE = (320,180)  # Camera capture size
PROCESS_SIZE = (320, 180)  # Processing size
MAX_MASKS = 3  # Maximum number of masks to process
VISUALIZATION_SCALE = 1.0  # No downscaling for visualization
POINTS_PER_SIDE = 8  # Points for coverage
TARGET_FPS = 20  # Target FPS for processing
DEBUG_FEATURES = True  # Enable feature point visualization

# Feature detection parameters
FEATURE_PARAMS = dict(
    maxCorners=300,       # Number of points to detect
    qualityLevel=0.1,     # Minimum quality threshold (lower means more points)
    minDistance=3,        # Minimum distance between points
    blockSize=5           # Size of blocks for corner detection
)

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),     # Size of search window for each pyramid level
    maxLevel=3,           # Number of pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Motion tracking parameters
MIN_MOTION_DISTANCE = 0.5   # Minimum distance to consider as motion
MOTION_DECAY = 0.8         # How quickly motion fades out
MOTION_THRESHOLD = 0.1     # Threshold for significant motion

# Performance monitoring constants
PERF_WINDOW_SIZE = 100
LOG_INTERVAL = 5.0

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

# Motion tracking state
prev_gray = None
prev_points = None
motion_history = None

def create_debug_frame(frame, curr_points, prev_points, motion_vectors=None, motion_areas=None):
    """
    Create a debug visualization showing feature detection and tracking.
    
    Args:
        frame: Original frame
        curr_points: Current detected feature points
        prev_points: Previous frame's feature points
        motion_vectors: Optional motion vector information
        motion_areas: Optional motion area contours
    
    Returns:
        Frame with debug visualization overlaid
    """
    debug_frame = frame.copy()
    
    # Draw all detected feature points
    if curr_points is not None:
        for point in curr_points:
            x, y = point.ravel()
            # Green circle for current points
            cv2.circle(debug_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # Draw motion tracking information
    if prev_points is not None and motion_vectors is not None:
        for (curr, prev, motion) in zip(curr_points, prev_points, motion_vectors):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            
            # Only draw significant motion
            if np.sqrt(motion[0]**2 + motion[1]**2) > MIN_MOTION_DISTANCE:
                # Red circle for previous position
                cv2.circle(debug_frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
                # Yellow line showing motion vector
                cv2.line(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (0, 255, 255), 1)
    
    # Draw motion areas if available
    if motion_areas is not None:
        # Blue contours for motion areas
        cv2.drawContours(debug_frame, motion_areas, -1, (255, 0, 0), 1)
    
    # Add feature count to debug frame
    if curr_points is not None:
        cv2.putText(debug_frame, f"Features: {len(curr_points)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return debug_frame

def detect_motion(current_frame, previous_frame):
    """
    Detect motion using feature tracking and optical flow.
    Now includes visual debugging of feature detection and tracking.
    
    Args:
        current_frame: Current video frame
        previous_frame: Previous video frame
    
    Returns:
        tuple: (motion_contours, debug_frame, motion_intensity)
        - motion_contours contains areas of detected motion
        - debug_frame shows feature detection visualization
        - motion_intensity indicates overall amount of motion
    """
    global prev_gray, prev_points, motion_history
    
    if previous_frame is None:
        return None, None, 0.0

    try:
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize or update feature points
        if prev_gray is None:
            prev_gray = curr_gray.copy()
            prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
            motion_history = np.zeros_like(curr_gray, dtype=np.float32)
            return None, current_frame, 0.0
        
        # Detect new feature points if needed
        if prev_points is None or len(prev_points) < 10:  # Minimum number of points
            prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
            if prev_points is None:
                return None, current_frame, 0.0
        
        # Calculate optical flow
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **LK_PARAMS
        )
        
        # Filter valid points
        valid_mask = status.ravel() == 1
        if not np.any(valid_mask):
            return None, current_frame, 0.0
            
        curr_valid = curr_points[valid_mask]
        prev_valid = prev_points[valid_mask]
        
        # Calculate motion vectors
        motion_vectors = curr_valid - prev_valid
        distances = np.sqrt(np.sum(motion_vectors**2, axis=1))
        
        # Create motion contours for significant motion
        motion_contours = []
        significant_motion = distances > MIN_MOTION_DISTANCE
        if np.any(significant_motion):
            # Create point pairs for contour creation
            moving_curr = curr_valid[significant_motion]
            moving_prev = prev_valid[significant_motion]
            
            # Convert points to contour format
            if len(moving_curr) >= 3:  # Need at least 3 points for a contour
                hull = cv2.convexHull(moving_curr.astype(np.float32))
                if hull is not None:
                    motion_contours = [hull.astype(np.int32)]
        
        # Create debug visualization
        debug_frame = create_debug_frame(
            current_frame, 
            curr_valid, 
            prev_valid, 
            motion_vectors,
            motion_contours
        )
        
        # Update previous frame state
        prev_gray = curr_gray.copy()
        prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
        
        # Calculate motion intensity
        motion_intensity = np.mean(distances) if len(distances) > 0 else 0.0
        
        return motion_contours, debug_frame, motion_intensity

    except Exception as e:
        print(f"Error in motion detection: {str(e)}")
        traceback.print_exc()
        return None, current_frame, 0.0

def compare_masks(mask1, mask2):
    """
    Compare two masks and return a similarity score.
    Score is based on Intersection over Union (IoU).
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    try:
        # Ensure masks are boolean arrays
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        # Calculate IoU
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)
        
        if union_sum == 0:
            return 0.0
            
        return intersection_sum / union_sum
    except Exception as e:
        print(f"Error comparing masks: {e}")
        return 0.0

def update_mask_tracking(current_masks):
    """
    Update mask tracking statistics and flags.
    
    Args:
        current_masks: List/array of current mask arrays
    """
    try:
        if current_masks is None:
            return
            
        # Initialize tracking arrays if needed
        if len(state.mask_stability_counters) != len(current_masks):
            state.mask_stability_counters = [0] * len(current_masks)
        if len(state.mask_flags) != len(current_masks):
            state.mask_flags = ['dynamic'] * len(current_masks)
            
        if state.previous_masks is None:
            state.previous_masks = current_masks
            state.mask_change_scores = [0.0] * len(current_masks)
            return
            
        # Ensure we have the same number of masks to compare
        min_masks = min(len(current_masks), len(state.previous_masks))
        new_change_scores = []
        
        # Compare each mask with its previous version
        for i in range(min_masks):
            similarity = compare_masks(current_masks[i], state.previous_masks[i])
            new_change_scores.append(similarity)
            
            # Update stability counter
            if similarity >= state.STABILITY_THRESHOLD:
                state.mask_stability_counters[i] += 1
            else:
                state.mask_stability_counters[i] = 0
                
            # Update mask flag based on stability
            if state.mask_stability_counters[i] >= state.STABILITY_COUNT_THRESHOLD:
                state.mask_flags[i] = 'stable'
            else:
                state.mask_flags[i] = 'dynamic'
                
        # Update tracking state
        state.mask_change_scores = new_change_scores
        state.previous_masks = current_masks.copy()
        
    except Exception as e:
        print(f"Error updating mask tracking: {e}")
        traceback.print_exc()

def generate_vibrant_colors(n):
    """
    Generate n vibrant, visually distinct colors.
    
    Uses HSV color space with golden ratio spacing for hue values,
    and random variation in saturation and value for visual interest.
    """
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

def get_cpu_temp():
    """Read CPU temperature from system file."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000
        return temp
    except:
        return 0.0

def log_performance_stats():
    """Log various performance statistics periodically."""
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
    print(f"CPU Temperature: {stats.get('cpu_temps', {}).get('mean', 0):.1f}°C")
    if 'frame_intervals' in stats:
        print(f"Frame Intervals: {stats['frame_intervals']['mean']*1000:.1f}ms (FPS: {1/stats['frame_intervals']['mean']:.1f})")
    print("\nProcessing Times (ms):")
    print(f"Resize:        {stats.get('resize_times', {}).get('mean', 0)*1000:.1f}")
    print(f"Embedding:     {stats.get('embedding_times', {}).get('mean', 0)*1000:.1f}")
    print(f"Inference:     {stats.get('inference_times', {}).get('mean', 0)*1000:.1f}")
    print(f"Visualization: {stats.get('visualization_times', {}).get('mean', 0)*1000:.1f}")
    print(f"Total:         {stats.get('total_process_times', {}).get('mean', 0)*1000:.1f}")
    print("===========================\n")
    
    last_log_time = current_time

def encode_frame_to_base64(frame):
    """Convert a frame to base64 encoded JPEG for streaming."""
    if frame is None:
        return None
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

def generate_visualization(frame, masks, scores):
    """
    Generate visualization of masks overlaid on the frame.
    
    Args:
        frame: Original frame to overlay masks on
        masks: List of boolean masks
        scores: Confidence scores for each mask
        
    Returns:
        Frame with visualized masks
    """
    t_start = time.time()
    # Always start with a copy of the original frame
    result = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    try:
        if masks is not None and len(masks) > 0:
            print(f"Visualizing {len(masks)} masks")
            colors = generate_vibrant_colors(len(masks))

            if scores is not None:
                mask_score_pairs = list(zip(masks, scores, colors, state.mask_flags))
                mask_score_pairs.sort(key=lambda x: x[1], reverse=True)
                masks, scores, colors, flags = zip(*mask_score_pairs)

            for i, (mask, color, flag) in enumerate(zip(masks, colors, state.mask_flags)):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                # For stable masks, convert to black and white
                if flag == 'stable':
                    overlay = np.zeros_like(frame)
                    overlay[mask_resized] = [255, 255, 255]  # White
                else:
                    overlay = np.zeros_like(frame)
                    overlay[mask_resized] = color

                base_alpha = 0.25 + (0.15 * (i / len(masks)))
                result = cv2.addWeighted(result, 1 - base_alpha, overlay, base_alpha, 0)

                mask_uint8 = mask_resized.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                thickness = 1  # Reduced thickness for all contours
                contour_color = [0, 0, 0] if flag == 'stable' else color.tolist()  # Black for stable masks
                cv2.drawContours(result, contours, -1, contour_color, thickness)
        else:
            print("No masks available for visualization")

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
        return frame

    perf_stats['visualization_times'].append(time.time() - t_start)
    return result

def cleanup():
    """Cleanup resources when shutting down."""
    if state.picam2 is not None:
        try:
            state.picam2.stop()
        except:
            pass