import numpy as np
import cv2
from mask_types import PhysicsMask
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Physics constants - adjusted for more responsive motion
VELOCITY_DAMPING = 0.85  # More damping (was 0.92)
MINIMUM_FORCE = 0.3      # Lower minimum force for easier triggering
FORCE_SCALE = 4.0       # Increased force scale (was 2.5)
MOTION_AREA_THRESHOLD = 0.05  # Lower threshold for motion detection (was 0.15)
EDGE_BUFFER = 5         # Buffer zone around mask bounds
VELOCITY_MIN = 0.1      # Minimum velocity threshold
MAX_VELOCITY = 25.0     # Increased max velocity (was 15.0)

def detect_motion_collision(mask: PhysicsMask, flow: np.ndarray, 
                          magnitude: np.ndarray, angle: np.ndarray,
                          threshold: float = 0.45) -> Tuple[float, float]:  # Lower threshold for easier triggering
    """
    Enhanced motion collision detection that better handles independent mask physics.
    Returns forces to apply to the mask based on detected motion.
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
    
    # Calculate weighted average force with increased influence
    avg_force_x = np.average(flow_x, weights=magnitude_weights)
    avg_force_y = np.average(flow_y, weights=magnitude_weights)
    
    # Scale force by overlap ratio and mass with increased effect
    force_x = (avg_force_x * FORCE_SCALE * overlap_ratio) / mask.mass
    force_y = (avg_force_y * FORCE_SCALE * overlap_ratio) / mask.mass
    
    # Print debug info for significant forces
    if abs(force_x) > MINIMUM_FORCE or abs(force_y) > MINIMUM_FORCE:
        print(f"\nMotion collision detected:")
        print(f"  Forces: ({force_x:.2f}, {force_y:.2f})")
        print(f"  Overlap ratio: {overlap_ratio:.2f}")
        print(f"  Magnitude mean: {np.mean(magnitude_weights):.2f}")
    
    return force_x, force_y

def update_mask_physics(mask: PhysicsMask, frame_size: Tuple[int, int]) -> None:
    """
    Update mask physics with improved velocity handling and boundary checking.
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
    
    # Check if mask is leaving the frame - be more aggressive about cleanup
    margin = max(mask_width, mask_height) // 2
    if (new_x + margin < 0 or new_x - margin > width or 
        new_y + margin < 0 or new_y - margin > height):
        print(f"Mask leaving frame at position ({new_x}, {new_y})")
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
    if not masks:
        return

    num_masks = len(masks)
    if num_masks < 2:
        return

    for i in range(num_masks):
        # Ensure i is an integer
        i = int(i)
        if not (0 <= i < num_masks):
            continue

        mask1 = masks[i]
        if not mask1.is_active:
            continue

        for j in range(i + 1, num_masks):
            # Ensure j is an integer
            j = int(j)
            if not (0 <= j < num_masks):
                continue

            mask2 = masks[j]
            if not mask2.is_active:
                continue

            if check_mask_collision(mask1, mask2):
                # Calculate collision normal
                dx = float(mask2.position[0] - mask1.position[0])
                dy = float(mask2.position[1] - mask1.position[1])
                distance = float(np.sqrt(dx * dx + dy * dy))
                
                if distance < 0.0001:  # Avoid division by zero
                    continue
                    
                # Normalize collision vector
                nx = dx / distance
                ny = dy / distance
                
                # Calculate relative velocity
                rvx = float(mask1.dx - mask2.dx)
                rvy = float(mask1.dy - mask2.dy)
                
                # Calculate relative velocity along normal
                rel_vel_normal = float(rvx * nx + rvy * ny)
                
                # Do not resolve if objects are separating
                if rel_vel_normal > 0:
                    continue
                
                # Coefficient of restitution (bounciness)
                restitution = 0.8
                
                # Calculate collision impulse
                j = -(1.0 + restitution) * rel_vel_normal
                total_inverse_mass = float(1.0/mask1.mass + 1.0/mask2.mass)
                if total_inverse_mass <= 0:
                    continue
                    
                j /= total_inverse_mass
                
                # Apply impulse to both masks
                impulse_x1 = float(-(j * nx) / mask1.mass)
                impulse_y1 = float(-(j * ny) / mask1.mass)
                impulse_x2 = float((j * nx) / mask2.mass)
                impulse_y2 = float((j * ny) / mask2.mass)
                
                mask1.dx += impulse_x1
                mask1.dy += impulse_y1
                mask2.dx += impulse_x2
                mask2.dy += impulse_y2
                
                # Ensure velocities are within bounds
                for mask in [mask1, mask2]:
                    mask.dx = float(np.clip(mask.dx, -MAX_VELOCITY, MAX_VELOCITY))
                    mask.dy = float(np.clip(mask.dy, -MAX_VELOCITY, MAX_VELOCITY))

def calculate_mask_energy(mask: PhysicsMask) -> float:
    """
    Calculate the total energy (kinetic + potential) of a mask.
    
    Args:
        mask: PhysicsMask object
        
    Returns:
        float: Total energy of the mask
    """
    # Calculate kinetic energy (KE = 1/2 * m * v^2)
    velocity_squared = float(mask.dx * mask.dx + mask.dy * mask.dy)
    kinetic_energy = 0.5 * float(mask.mass) * velocity_squared
    
    # Calculate potential energy (simplified, based on height)
    potential_energy = float(mask.mass) * 9.81 * float(mask.position[1])
    
    return float(kinetic_energy + potential_energy)