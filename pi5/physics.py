import numpy as np
import cv2
from mask_types import PhysicsMask
from typing import Tuple, Optional

def detect_motion_collision(mask: PhysicsMask, flow: np.ndarray, 
                          magnitude: np.ndarray, angle: np.ndarray,
                          threshold: float = 0.65) -> Tuple[float, float]:  # Increased threshold
    """
    Detect collision between motion and a specific mask, returning force vectors.
    Now with enhanced individual mask collision detection and higher threshold.
    
    Args:
        mask: PhysicsMask object representing the single mask we're checking
        flow: Dense optical flow data containing motion vectors
        magnitude: Motion magnitude array (how strong the motion is at each point)
        angle: Motion angle array (direction of motion at each point)
        threshold: Minimum motion magnitude to consider (increased to reduce sensitivity)
        
    Returns:
        Tuple of (dx, dy) representing the force to apply to this specific mask
    """
    # Get the bounding box of just this mask
    x1, y1, x2, y2 = mask.bounds
    
    # Add a small margin around the bounds to better detect approaching motion
    margin = 5
    h, w = magnitude.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)
    
    # Extract the relevant regions from our data arrays
    mask_region = mask.mask[y1:y2+1, x1:x2+1]
    flow_region = flow[y1:y2+1, x1:x2+1]
    magnitude_region = magnitude[y1:y2+1, x1:x2+1]
    
    # Calculate the percentage of the mask that has significant motion
    significant_motion = magnitude_region > threshold
    mask_area = np.sum(mask_region)
    if mask_area == 0:
        return 0.0, 0.0
        
    motion_overlap = np.sum(significant_motion & mask_region)
    motion_percentage = motion_overlap / mask_area
    
    # Only respond if a significant portion of the mask sees motion
    MOTION_AREA_THRESHOLD = 0.2  # 20% of the mask must see motion
    if motion_percentage < MOTION_AREA_THRESHOLD:
        return 0.0, 0.0
    
    # Get motion vectors where we have both significant motion and mask overlap
    flow_x = flow_region[..., 0][significant_motion & mask_region]
    flow_y = flow_region[..., 1][significant_motion & mask_region]
    
    if len(flow_x) == 0 or len(flow_y) == 0:
        return 0.0, 0.0
    
    # Calculate average motion vector and scale by the overlap percentage
    # This makes the force stronger when more of the mask is affected
    force_x = np.mean(flow_x) * motion_percentage
    force_y = np.mean(flow_y) * motion_percentage
    
    # Scale the forces by mass
    return force_x / mask.mass, force_y / mask.mass

def update_mask_physics(mask: PhysicsMask, frame_size: Tuple[int, int]) -> None:
    """
    Update a mask's position based on its physics properties.
    Added velocity dampening for more controlled movement.
    """
    height, width = frame_size
    
    # Add velocity dampening
    VELOCITY_DAMPING = 0.8  # Reduces "slipperiness"
    
    # Update position based on current velocity
    x, y = mask.position
    new_x = x + int(mask.dx * VELOCITY_DAMPING)
    new_y = y + int(mask.dy * VELOCITY_DAMPING)
    
    # Apply friction to gradually reduce velocity
    mask.dx *= mask.friction
    mask.dy *= mask.friction
    
    # Add minimum velocity threshold to stop tiny movements
    MIN_VELOCITY = 0.1
    if abs(mask.dx) < MIN_VELOCITY: mask.dx = 0
    if abs(mask.dy) < MIN_VELOCITY: mask.dy = 0
    
    # Check if mask is completely off screen
    x1, y1, x2, y2 = mask.bounds
    mask_width = x2 - x1
    mask_height = y2 - y1
    
    if (new_x + mask_width < 0 or new_x > width or 
        new_y + mask_height < 0 or new_y > height):
        mask.is_active = False
    else:
        mask.position = (new_x, new_y)

def translate_mask(mask: np.ndarray, dx: int, dy: int, 
                  frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Translate a mask by (dx, dy) pixels while keeping it within frame bounds.
    Now includes bounds checking for safer operation.
    """
    height, width = frame_size
    
    # Ensure translation stays within bounds
    if dx > width or dx < -width or dy > height or dy < -height:
        return np.zeros_like(mask, dtype=bool)
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), M, (width, height)) > 0