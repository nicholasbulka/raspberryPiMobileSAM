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
INPUT_SIZE = (320,180)  # Camera capture size (increased)
PROCESS_SIZE = (320, 180)  # SAM processing size (unchanged)
MAX_MASKS = 3  # Maximum number of masks to process
VISUALIZATION_SCALE = 1.0  # No downscaling for visualization
POINTS_PER_SIDE = 8  # Increased points for better coverage
TARGET_FPS = 20  # Target FPS for processing

# Motion detection constants
FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
MIN_MOTION_DISTANCE = 0.5  # Reduced: Minimum pixel distance to consider as motion
MOTION_BLUR_SIZE = 31   # Increased: Size of Gaussian blur for motion mask
MOTION_THRESHOLD = 0.05  # Reduced: Threshold for motion intensity

# Performance monitoring constants
PERF_WINDOW_SIZE = 100  # Number of frames to keep stats for
LOG_INTERVAL = 5.0  # Seconds between logging performance stats

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

def detect_motion(current_frame, previous_frame):
    """
    Detect motion using optical flow and feature tracking.
    Returns a motion mask and motion intensity.
    """
    if previous_frame is None:
        return None, 0.0

    try:
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Detect good features to track
        prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        if prev_points is None:
            return None, 0.0

        # Calculate optical flow
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **LK_PARAMS
        )

        # Filter out points where flow wasn't found
        good_curr = curr_points[status == 1]
        good_prev = prev_points[status == 1]

        # Calculate motion distances
        if len(good_curr) == 0 or len(good_prev) == 0:
            return None, 0.0

        # Calculate motion distances and normalize them
        motion_distances = np.sqrt(np.sum((good_curr - good_prev) ** 2, axis=1))
        max_distance = np.max(motion_distances)
        if max_distance > 0:
            normalized_distances = motion_distances / max_distance
        else:
            return None, 0.0

        # Create motion mask with continuous values
        motion_mask = np.zeros_like(curr_gray, dtype=np.float32)
        
        # Add motion points to mask with distance-based intensity
        for (x, y), distance in zip(good_curr, normalized_distances):
            # Use the normalized distance as intensity
            cv2.circle(
                motion_mask, 
                (int(x), int(y)), 
                int(MOTION_BLUR_SIZE/2),  # Increased radius for better coverage
                distance,  # Use normalized distance as intensity
                -1  # Fill the circle
            )

        # Apply Gaussian blur for smooth transitions
        motion_mask = cv2.GaussianBlur(motion_mask, (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0)
        
        # Normalize the blurred mask
        motion_mask = cv2.normalize(motion_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply threshold but maintain float values
        motion_mask = np.where(motion_mask > MOTION_THRESHOLD, motion_mask, 0)

        # Calculate overall motion intensity as mean of significant motion
        motion_intensity = np.mean(motion_distances)

        return motion_mask, motion_intensity

    except Exception as e:
        print(f"Error in motion detection: {str(e)}")
        traceback.print_exc()
        return None, 0.0

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