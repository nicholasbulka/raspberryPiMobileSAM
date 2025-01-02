import cv2
import base64
import time
import statistics
from collections import deque
import traceback
import shared_state as state

# Performance monitoring configuration
PERF_WINDOW_SIZE = 100
LOG_INTERVAL = 5.0

# Initialize performance tracking
perf_stats = {
    'resize_times': deque(maxlen=PERF_WINDOW_SIZE),
    'flow_times': deque(maxlen=PERF_WINDOW_SIZE),
    'visualization_times': deque(maxlen=PERF_WINDOW_SIZE),
    'frame_intervals': deque(maxlen=PERF_WINDOW_SIZE),
    'cpu_temps': deque(maxlen=PERF_WINDOW_SIZE),
    'total_process_times': deque(maxlen=PERF_WINDOW_SIZE),
    'embedding_times': deque(maxlen=PERF_WINDOW_SIZE),
    'inference_times': deque(maxlen=PERF_WINDOW_SIZE)
}

# Global state for performance monitoring
last_log_time = time.time()

def encode_frame_to_base64(frame):
    """
    Convert a video frame to a base64-encoded JPEG string for web streaming.
    Uses moderate JPEG compression to balance quality and bandwidth.
    
    Args:
        frame: OpenCV BGR image array
        
    Returns:
        str: Base64-encoded JPEG string, or None if frame is invalid
    """
    if frame is None:
        return None
        
    try:
        # Encode frame as JPEG with quality=70 (good balance of quality vs size)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        # Convert to base64 string
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {str(e)}")
        return None

def get_cpu_temp():
    """
    Read CPU temperature from system file.
    Used for monitoring system performance and thermal state.
    
    Returns:
        float: CPU temperature in Celsius, or 0.0 if reading fails
    """
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            # Temperature is reported in millidegrees Celsius
            temp = float(f.read().strip()) / 1000
        return temp
    except Exception as e:
        print(f"Error reading CPU temperature: {str(e)}")
        return 0.0

def log_performance_stats():
    """
    Log various performance statistics periodically.
    Provides insights into system performance and processing times.
    """
    global last_log_time
    current_time = time.time()
    
    # Only log at specified intervals to avoid spam
    if current_time - last_log_time < LOG_INTERVAL:
        return
    
    try:
        # Calculate statistics for each metric
        stats = {}
        for key, values in perf_stats.items():
            if values:
                stats[key] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        # Print formatted statistics
        print("\n=== Performance Statistics ===")
        print(f"CPU Temperature: {stats.get('cpu_temps', {}).get('mean', 0):.1f}°C")
        
        if 'frame_intervals' in stats:
            fps = 1/stats['frame_intervals']['mean']
            print(f"Frame Intervals: {stats['frame_intervals']['mean']*1000:.1f}ms "
                  f"(FPS: {fps:.1f})")
        
        print("\nProcessing Times (ms):")
        print(f"Flow:          {stats.get('flow_times', {}).get('mean', 0)*1000:.1f}")
        print(f"Visualization: {stats.get('visualization_times', {}).get('mean', 0)*1000:.1f}")
        print("===========================\n")
        
        last_log_time = current_time
        
    except Exception as e:
        print(f"Error logging performance stats: {str(e)}")

def cleanup():
    """
    Clean up system resources before shutdown.
    Ensures camera and other resources are properly released.
    """
    if state.picam2 is not None:
        try:
            state.picam2.stop()
            print("Camera stopped successfully")
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")
            
def update_performance_stats(stat_name, value):
    """
    Update a specific performance statistic.
    Provides a thread-safe way to track performance metrics.
    
    Args:
        stat_name: Name of the statistic to update
        value: New value to add to the statistic
    """
    if stat_name in perf_stats:
        perf_stats[stat_name].append(value)
    else:
        print(f"Warning: Unknown performance stat '{stat_name}'")