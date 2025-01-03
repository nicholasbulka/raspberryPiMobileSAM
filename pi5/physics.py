import numpy as np
import cv2
from mask_types import PhysicsMask
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Physics constants
VELOCITY_DAMPING = 0.92  # Higher value = less damping
MINIMUM_FORCE = 0.5      # Minimum force to apply
FORCE_SCALE = 2.5       # Scale factor for motion forces
MOTION_AREA_THRESHOLD = 0.15  # Percentage of mask that must see motion
EDGE_BUFFER = 5         # Buffer zone around mask bounds
VELOCITY_MIN = 0.1      # Minimum velocity threshold
MAX_VELOCITY = 15.0     # Maximum velocity cap

def detect_motion_collision(mask: PhysicsMask, flow: np.ndarray, 
                          magnitude: np.ndarray, angle: np.ndarray,
                          threshold: float = 0.65) -> Tuple[float, float]:
    """
    Enhanced motion collision detection that better handles independent mask physics.
    Returns forces to apply to the mask based on detected motion.
    
    Args:
        mask: PhysicsMask object for collision detection
        flow: Dense optical flow data
        magnitude: Motion magnitude array
        angle: Motion angle array
        threshold: Motion detection threshold
        
    Returns:
        Tuple of (dx, dy) forces to apply
    """
    # Get mask bounds with buffer zone
    x1, y1, x2, y2 = mask.bounds
    h, w = magnitude.shape[:2]
    
    # Add buffer zone while keeping within frame
    x1 = max(0, x1 - EDGE_BUFFER)
    y1 = max(0, y1 - EDGE_BUFFER)
    x2 = min(w - 1, x2 + EDGE_BUFFER)
    y2 = min(h - 1, y2 + EDGE_BUFFER)
    
    # Extract regions of interest
    mask_region = mask.mask[y1:y2+1, x1:x2+1]
    flow_region = flow[y1:y2+1, x1:x2+1]
    magnitude_region = magnitude[y1:y2+1, x1:x2+1]
    
    # Find significant motion areas
    motion_mask = magnitude_region > threshold
    mask_overlap = motion_mask & mask_region
    
    # Calculate percentage of mask affected by motion
    mask_area = np.sum(mask_region)
    if mask_area == 0:
        return 0.0, 0.0
        
    overlap_ratio = np.sum(mask_overlap) / mask_area
    if overlap_ratio < MOTION_AREA_THRESHOLD:
        return 0.0, 0.0
    
    # Calculate weighted average motion vector
    flow_x = flow_region[..., 0][mask_overlap]
    flow_y = flow_region[..., 1][mask_overlap]
    magnitude_weights = magnitude_region[mask_overlap]
    
    if len(flow_x) == 0 or len(flow_y) == 0:
        return 0.0, 0.0
    
    # Calculate weighted average force
    avg_force_x = np.average(flow_x, weights=magnitude_weights)
    avg_force_y = np.average(flow_y, weights=magnitude_weights)
    
    # Scale force by overlap ratio and mass
    force_x = (avg_force_x * FORCE_SCALE * overlap_ratio) / mask.mass
    force_y = (avg_force_y * FORCE_SCALE * overlap_ratio) / mask.mass
    
    # Apply minimum force threshold
    if abs(force_x) < MINIMUM_FORCE:
        force_x = 0
    if abs(force_y) < MINIMUM_FORCE:
        force_y = 0
        
    return force_x, force_y

def update_mask_physics(mask: PhysicsMask, frame_size: Tuple[int, int]) -> None:
    """
    Update mask physics with improved velocity handling and boundary checking.
    
    Args:
        mask: PhysicsMask object to update
        frame_size: (height, width) of the frame
    """
    height, width = frame_size
    
    # Apply velocity changes
    x, y = mask.position
    
    # Update velocities with damping
    mask.dx *= VELOCITY_DAMPING
    mask.dy *= VELOCITY_DAMPING
    
    # Cap maximum velocity
    mask.dx = np.clip(mask.dx, -MAX_VELOCITY, MAX_VELOCITY)
    mask.dy = np.clip(mask.dy, -MAX_VELOCITY, MAX_VELOCITY)
    
    # Stop tiny movements
    if abs(mask.dx) < VELOCITY_MIN:
        mask.dx = 0
    if abs(mask.dy) < VELOCITY_MIN:
        mask.dy = 0
    
    # Calculate new position
    new_x = x + int(mask.dx)
    new_y = y + int(mask.dy)
    
    # Get mask dimensions
    x1, y1, x2, y2 = mask.bounds
    mask_width = x2 - x1
    mask_height = y2 - y1
    
    # Check if mask is leaving the frame
    if (new_x + mask_width < 0 or new_x > width or 
        new_y + mask_height < 0 or new_y > height):
        mask.is_active = False
    else:
        # Update position
        mask.position = (new_x, new_y)
        
        # Apply friction
        mask.dx *= mask.friction
        mask.dy *= mask.friction

def translate_mask(mask: np.ndarray, dx: int, dy: int, 
                  frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Translate a mask by (dx, dy) pixels while keeping it within frame bounds.
    Now with improved boundary handling and smooth translation.
    
    Args:
        mask: Binary mask array
        dx: Horizontal translation
        dy: Vertical translation
        frame_size: (height, width) of frame
        
    Returns:
        Translated binary mask array
    """
    height, width = frame_size
    
    # Create larger translation matrix for smooth movement
    if abs(dx) > width or abs(dy) > height:
        return np.zeros_like(mask, dtype=bool)
    
    # Use affine transformation for smooth translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return translated > 0

def check_mask_collision(mask1: PhysicsMask, mask2: PhysicsMask) -> bool:
    """
    Check if two masks are colliding.
    
    Args:
        mask1: First PhysicsMask object
        mask2: Second PhysicsMask object
        
    Returns:
        True if masks are colliding, False otherwise
    """
    # Get mask bounds
    x1a, y1a, x2a, y2a = mask1.bounds
    x1b, y1b, x2b, y2b = mask2.bounds
    
    # Check for bounding box overlap
    if (x1a > x2b or x2a < x1b or
        y1a > y2b or y2a < y1b):
        return False
    
    # Check for actual mask overlap
    intersection = np.logical_and(mask1.mask, mask2.mask)
    return np.any(intersection)

def resolve_mask_collisions(masks: List[PhysicsMask]) -> None:
    """
    Resolve collisions between all masks using elastic collision physics.
    
    Args:
        masks: List of PhysicsMask objects
    """
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if not masks[i].is_active or not masks[j].is_active:
                continue
                
            if check_mask_collision(masks[i], masks[j]):
                # Calculate collision normal
                dx = masks[j].position[0] - masks[i].position[0]
                dy = masks[j].position[1] - masks[i].position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                
                if distance == 0:  # Avoid division by zero
                    continue
                    
                # Normalize collision vector
                nx = dx / distance
                ny = dy / distance
                
                # Calculate relative velocity
                rvx = masks[i].dx - masks[j].dx
                rvy = masks[i].dy - masks[j].dy
                
                # Calculate relative velocity along normal
                rel_vel_normal = rvx * nx + rvy * ny
                
                # Do not resolve if objects are separating
                if rel_vel_normal > 0:
                    continue
                
                # Coefficient of restitution (bounciness)
                restitution = 0.8
                
                # Calculate collision impulse
                j = -(1 + restitution) * rel_vel_normal
                j /= 1/masks[i].mass + 1/masks[j].mass
                
                # Apply impulse to both masks
                masks[i].dx -= (j * nx) / masks[i].mass
                masks[i].dy -= (j * ny) / masks[i].mass
                masks[j].dx += (j * nx) / masks[j].mass
                masks[j].dy += (j * ny) / masks[j].mass
                
                # Ensure velocities are within bounds
                for mask in [masks[i], masks[j]]:
                    mask.dx = np.clip(mask.dx, -MAX_VELOCITY, MAX_VELOCITY)
                    mask.dy = np.clip(mask.dy, -MAX_VELOCITY, MAX_VELOCITY)

def calculate_mask_energy(mask: PhysicsMask) -> float:
    """
    Calculate the total energy (kinetic + potential) of a mask.
    
    Args:
        mask: PhysicsMask object
        
    Returns:
        float: Total energy of the mask
    """
    # Calculate kinetic energy (KE = 1/2 * m * v^2)
    velocity_squared = mask.dx * mask.dx + mask.dy * mask.dy
    kinetic_energy = 0.5 * mask.mass * velocity_squared
    
    # Calculate potential energy (simplified, based on height)
    potential_energy = mask.mass * 9.81 * mask.position[1]  # g = 9.81 m/s^2
    
    return kinetic_energy + potential_energy