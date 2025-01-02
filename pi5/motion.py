import cv2
import numpy as np
import time
import traceback
from colors import color_scheme
from mask import resize_mask
from utils import update_performance_stats
import shared_state as state

# Dense optical flow parameters for Farneback algorithm
FLOW_PARAMS = {
    'pyr_scale': 0.5,     # Pyramid scale between levels
    'levels': 3,          # Number of pyramid levels
    'winsize': 15,        # Size of window for flow calculation
    'iterations': 3,      # Number of iterations at each level
    'poly_n': 5,          # Size of pixel neighborhood
    'poly_sigma': 1.2,    # Standard deviation of Gaussian
    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
}

# Motion effect parameters
MOTION_THRESHOLD = 0.2       # Minimum motion magnitude for effect
EFFECT_PULSE_RATE = 5.0      # Rate of effect pulsing (Hz)
SPARKLE_PROBABILITY = 0.1    # Base probability for sparkle effect
DISPLACEMENT_SCALE = 0.5     # Scale factor for motion displacement

def detect_motion(current_frame, previous_frame):
    """
    Detect motion using dense optical flow and return motion information in the 
    format expected by the processing pipeline. This function bridges our new
    dense flow implementation with the existing pipeline.
    
    Args:
        current_frame: Current video frame
        previous_frame: Previous video frame
        
    Returns:
        tuple: (motion_contours, debug_frame, motion_intensity)
    """
    # Get dense optical flow data
    flow, magnitude, angle, flow_vis, motion_intensity = detect_motion_dense(
        current_frame, previous_frame
    )
    
    if flow is None:
        return None, current_frame, 0.0
        
    try:
        # Create motion contours from magnitude
        motion_mask = magnitude > MOTION_THRESHOLD
        motion_mask = motion_mask.astype(np.uint8) * 255
        
        # Find contours in the motion mask
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Use flow visualization as debug frame
        debug_frame = flow_vis
        
        return contours, debug_frame, motion_intensity
        
    except Exception as e:
        print(f"Error creating motion contours: {str(e)}")
        traceback.print_exc()
        return None, current_frame, 0.0

def detect_motion_dense(current_frame, previous_frame):
    """
    Detect motion using dense optical flow. This function calculates motion 
    for every pixel in the frame using the Farneback algorithm, which provides
    rich motion information we can use for artistic effects.
    
    Args:
        current_frame: Current video frame
        previous_frame: Previous video frame
        
    Returns:
        tuple: (flow, magnitude, angle, flow_visualization, motion_intensity)
    """
    if previous_frame is None:
        return None, None, None, current_frame, 0.0
    
    try:
        t_start = time.time()
        
        # Convert frames to grayscale for flow calculation
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, 
            curr_gray, 
            None,
            **FLOW_PARAMS
        )
        
        # Calculate magnitude and angle of flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create artistic visualization
        flow_vis = create_flow_visualization(current_frame, magnitude, angle)
        
        # Calculate overall motion intensity
        motion_intensity = np.mean(magnitude)
        
        update_performance_stats('flow_times', time.time() - t_start)
        
        return flow, magnitude, angle, flow_vis, motion_intensity
        
    except Exception as e:
        print(f"Error in motion detection: {str(e)}")
        traceback.print_exc()
        return None, None, None, current_frame, 0.0

def create_flow_visualization(frame, magnitude, angle):
    """
    Create an artistic visualization of the optical flow field by mapping
    motion direction to color hue and motion magnitude to brightness.
    This provides an intuitive way to see how objects are moving in the scene.
    
    Args:
        frame: Original frame
        magnitude: Motion magnitude array
        angle: Motion angle array
    """
    # Create HSV representation of flow
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255  # Full saturation
    
    # Map angle to hue (convert to degrees, scale to 0-180)
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Map magnitude to value/brightness
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = normalized_magnitude
    
    # Convert to BGR and blend with original frame
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)
    
    return result

def apply_motion_effects(frame, masks, flow, magnitude, angle):
    """
    Apply artistic effects where motion intersects with masks. This function
    creates dynamic visual effects that respond to the motion in the scene.
    Effects include color pulsing, motion-based displacement, and sparkles.
    
    Args:
        frame: Original frame
        masks: List of mask arrays
        flow: Optical flow data
        magnitude: Motion magnitude array
        angle: Motion angle array
    """
    if masks is None or len(masks) == 0:
        return frame
    
    try:
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Create motion intensity map
        motion_intensity = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        time_factor = np.sin(time.time() * EFFECT_PULSE_RATE) * 0.5 + 0.5
        
        for mask_idx, mask in enumerate(masks):
            # Resize mask and find motion intersection
            mask_resized = resize_mask(mask, (width, height))
            intersection = mask_resized & (motion_intensity > MOTION_THRESHOLD)
            
            if np.any(intersection):
                # Generate effect color from our scheme
                effect_color = color_scheme.generate_from_base(
                    int(time.time() * 10) + mask_idx
                )
                
                # Apply time-based effects
                if time_factor > 0.7:
                    # Pulsing color effect
                    result[intersection] = effect_color
                elif time_factor > 0.3:
                    # Motion-based displacement
                    apply_displacement_effect(
                        result, frame, intersection, flow,
                        height, width
                    )
                else:
                    # Sparkle effect
                    apply_sparkle_effect(
                        result, intersection, magnitude
                    )
        
        return result
        
    except Exception as e:
        print(f"Error applying motion effects: {str(e)}")
        traceback.print_exc()
        return frame

def apply_displacement_effect(result, frame, intersection, flow, height, width):
    """
    Create a displacement effect by moving pixels based on the optical flow.
    This creates a dynamic warping effect in areas with motion.
    """
    y_coords, x_coords = np.where(intersection)
    for y, x in zip(y_coords, x_coords):
        # Calculate displacement based on flow
        flow_x = flow[y, x, 0]
        flow_y = flow[y, x, 1]
        new_x = int(x + flow_x * DISPLACEMENT_SCALE)
        new_y = int(y + flow_y * DISPLACEMENT_SCALE)
        
        # Ensure coordinates stay within frame bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Move pixels according to flow
        result[y, x] = frame[new_y, new_x]

def apply_sparkle_effect(result, intersection, magnitude):
    """
    Create a sparkle effect by randomly lighting up pixels based on 
    motion magnitude. Areas with more motion get more sparkles.
    """
    sparkle_prob = SPARKLE_PROBABILITY * magnitude[intersection]
    sparkle_mask = np.random.random(sparkle_prob.shape) < sparkle_prob
    result[intersection][sparkle_mask] = [255, 255, 255]