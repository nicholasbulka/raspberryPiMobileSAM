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
INPUT_SIZE = (320, 180)  # Camera capture size
PROCESS_SIZE = (160, 90)  # SAM processing size (increased)
MAX_MASKS = 3  # Maximum number of masks to process
VISUALIZATION_SCALE = 1.0  # No downscaling for visualization
POINTS_PER_SIDE = 4  # Increased points for better coverage
TARGET_FPS = 10  # Target FPS for processing

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

def generate_visualization(frame, masks, scores):
    t_start = time.time()
    # Always start with a copy of the original frame
    result = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    try:
        if masks is not None and len(masks) > 0:
            print(f"Visualizing {len(masks)} masks")
            # Create a single overlay for all masks
            overlay = np.zeros_like(frame, dtype=np.uint8)
            
            # Sort masks by score if available
            if scores is not None:
                print("scores were available")
                colors = generate_vibrant_colors(len(masks))
                mask_score_pairs = list(zip(masks, scores, colors))
                mask_score_pairs.sort(key=lambda x: x[1], reverse=True)
                masks, scores, colors = zip(*mask_score_pairs)
            else:
                print("scores were None")
                colors = generate_vibrant_colors(len(masks))

            for i, (mask, color) in enumerate(zip(masks, colors)):
                print(f"  Processing mask {i}: shape={mask.shape}, frame={frame.shape}, dtype={mask.dtype}")
                
                # Create colored mask
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[mask] = color
                
                # Add this mask to overlay with transparency
                cv2.addWeighted(overlay, 0.1, mask_color, 0.70, 0, overlay)
                
                # Draw contours for better visibility
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"    Found {len(contours)} contours")
                
                # Brighter contours for visibility
                contour_color = np.clip(color + np.array([80, 80, 80]), 0, 255).astype(np.uint8)
                cv2.drawContours(overlay, contours, -1, contour_color.tolist(), 2)

            # Blend the overlay with the original frame
            result = cv2.addWeighted(frame, 0.1, overlay, 0.7, 0)
            print("Visualization complete")

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
        return frame

    perf_stats['visualization_times'].append(time.time() - t_start)
    return result

def cleanup():
    if state.picam2 is not None:
        try:
            state.picam2.stop()
        except:
            pass

def get_change_bbox(current_frame, previous_frame, min_area=100, max_size=128):
    """
    Detect the bounding box of the largest changed region between frames.
    Returns: (x, y, w, h) or None if no significant change
    """
    if previous_frame is None:
        return None
        
    print(f"Frame shapes - Current: {current_frame.shape}, Previous: {previous_frame.shape}")
    print(f"Frame types - Current: {current_frame.dtype}, Previous: {previous_frame.dtype}")
    
    try:
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
    except Exception as e:
        print(f"Error in get_change_bbox: {str(e)}")
        print(f"Full details of frames:")
        print(f"Current frame: shape={current_frame.shape}, dtype={current_frame.dtype}, min={current_frame.min()}, max={current_frame.max()}")
        print(f"Previous frame: shape={previous_frame.shape}, dtype={previous_frame.dtype}, min={previous_frame.min()}, max={previous_frame.max()}")
        traceback.print_exc()
        return None

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
