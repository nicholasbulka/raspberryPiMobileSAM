import cv2
import numpy as np
import time
import traceback
from colors import color_scheme
from physics import (
    detect_motion_collision,
    update_mask_physics,
    translate_mask,
    resolve_mask_collisions,
    MOTION_AREA_THRESHOLD
)
from mask_types import PhysicsMask
from utils import update_performance_stats
import shared_state as state
from typing import List, Tuple, Optional

# Dense optical flow parameters
FLOW_PARAMS = {
    'pyr_scale': 0.5,     # Pyramid scale between levels
    'levels': 3,          # Number of pyramid levels
    'winsize': 15,        # Size of window for flow calculation
    'iterations': 3,      # Number of iterations at each level
    'poly_n': 5,          # Size of pixel neighborhood
    'poly_sigma': 1.2,    # Standard deviation of Gaussian
    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
}

# Motion visualization parameters
FLOW_SCALE = 1.5         # Scale factor for flow visualization
MIN_FLOW_MAGNITUDE = 0.5  # Minimum flow magnitude to visualize
FLOW_COLOR_SCALE = 15    # Scale factor for flow colors

def detect_motion(current_frame: np.ndarray, previous_frame: np.ndarray) -> Tuple[list, np.ndarray, float]:
    """
    Enhanced motion detection with improved physics handling.
    
    Args:
        current_frame: Current video frame
        previous_frame: Previous video frame
        
    Returns:
        Tuple of (motion contours, debug visualization, motion intensity)
    """
    # Get dense optical flow
    flow, magnitude, angle, flow_vis, motion_intensity = detect_motion_dense(
        current_frame,
        previous_frame
    )
    
    if flow is None:
        return None, current_frame, 0.0
    
    try:
        # Create motion mask
        motion_mask = magnitude > state.MIN_MOTION_THRESHOLD
        motion_mask = motion_mask.astype(np.uint8) * 255
        
        # Find motion contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Update physics for active masks
        frame_size = current_frame.shape[:2]
        active_masks = []
        
        with state.frame_lock:
            if state.current_masks:
                # Resolve collisions between masks first
                resolve_mask_collisions(state.current_masks)
                
                # Update each mask's physics
                for physics_mask in state.current_masks:
                    if not physics_mask.is_active:
                        continue
                    
                    # Calculate motion forces
                    dx, dy = detect_motion_collision(
                        physics_mask,
                        flow,
                        magnitude,
                        angle,
                        threshold=state.MIN_MOTION_THRESHOLD
                    )
                    
                    # Update velocities
                    physics_mask.dx += dx
                    physics_mask.dy += dy
                    
                    # Update position and check bounds
                    update_mask_physics(physics_mask, frame_size)
                    
                    # Translate mask if still active
                    if physics_mask.is_active:
                        physics_mask.mask = translate_mask(
                            physics_mask.mask,
                            int(physics_mask.dx),
                            int(physics_mask.dy),
                            frame_size
                        )
                        active_masks.append(physics_mask)
            
            # Update state with active masks
            state.current_masks = active_masks
        
        # Create debug visualization
        debug_frame = create_motion_visualization(
            current_frame,
            flow,
            magnitude,
            angle,
            motion_mask
        )
        
        return contours, debug_frame, motion_intensity
        
    except Exception as e:
        print(f"Error in motion detection: {str(e)}")
        traceback.print_exc()
        return None, current_frame, 0.0

def detect_motion_dense(current_frame: np.ndarray,
                       previous_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate dense optical flow between frames with improved accuracy.
    
    Args:
        current_frame: Current video frame
        previous_frame: Previous video frame
        
    Returns:
        Tuple of (flow, magnitude, angle, visualization, motion_intensity)
    """
    if previous_frame is None:
        return None, None, None, current_frame, 0.0
    
    try:
        t_start = time.time()
        
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            **FLOW_PARAMS
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True
        )
        
        # Create artistic flow visualization
        flow_vis = create_flow_visualization(
            current_frame,
            magnitude,
            angle
        )
        
        # Calculate motion intensity
        motion_intensity = np.mean(magnitude)
        
        update_performance_stats('flow_times', time.time() - t_start)
        
        return flow, magnitude, angle, flow_vis, motion_intensity
        
    except Exception as e:
        print(f"Error in dense motion detection: {str(e)}")
        traceback.print_exc()
        return None, None, None, current_frame, 0.0

def create_flow_visualization(frame: np.ndarray,
                            magnitude: np.ndarray,
                            angle: np.ndarray) -> np.ndarray:
    """
    Create enhanced visualization of optical flow field.
    
    Args:
        frame: Original video frame
        magnitude: Motion magnitude array
        angle: Motion angle array
        
    Returns:
        Flow visualization frame
    """
    # Create HSV representation
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    
    # Map angle to hue
    hsv[..., 0] = angle / 2.0
    
    # Map magnitude to value with enhanced contrast
    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    hsv[..., 2] = normalized_magnitude
    
    # Convert to BGR and blend with original
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add motion trails
    result = cv2.addWeighted(
        frame,
        0.7,
        flow_vis,
        0.3,
        0
    )
    
    return result

def create_motion_visualization(frame: np.ndarray,
                              flow: np.ndarray,
                              magnitude: np.ndarray,
                              angle: np.ndarray,
                              motion_mask: np.ndarray) -> np.ndarray:
    """
    Create comprehensive motion visualization with trails and vectors.
    
    Args:
        frame: Original video frame
        flow: Optical flow data
        magnitude: Motion magnitude array
        angle: Motion angle array
        motion_mask: Binary motion mask
        
    Returns:
        Debug visualization frame
    """
    # Start with flow visualization
    result = create_flow_visualization(frame, magnitude, angle)
    
    # Draw motion vectors
    step = 16  # Grid size for vector visualization
    h, w = frame.shape[:2]
    
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]
    
    # Filter out small movements
    mask = magnitude[y, x] > MIN_FLOW_MAGNITUDE
    
    # Draw arrows for significant motion
    for i, (start_point, fx_i, fy_i) in enumerate(zip(zip(x[mask], y[mask]),
                                                     fx[mask], fy[mask])):
        end_point = (int(start_point[0] + fx_i*FLOW_SCALE),
                    int(start_point[1] + fy_i*FLOW_SCALE))
        
        cv2.arrowedLine(
            result,
            (int(start_point[0]), int(start_point[1])),
            end_point,
            (0, 255, 0),
            1,
            tipLength=0.2
        )
    
    return result