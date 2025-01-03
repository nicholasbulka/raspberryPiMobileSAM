import cv2
import numpy as np
import traceback
from typing import List, Optional, Tuple
import shared_state as state
from colors import color_scheme
from mask_types import PhysicsMask
import time

def compare_masks(mask1: PhysicsMask, mask2: PhysicsMask) -> float:
    """
    Compare two masks using Intersection over Union (IoU).
    Now correctly handles PhysicsMask objects by accessing their numpy arrays.
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    try:
        # Access the underlying numpy arrays with .mask
        mask1_arr = mask1.mask.astype(bool)
        mask2_arr = mask2.mask.astype(bool)
        
        intersection = np.logical_and(mask1_arr, mask2_arr)
        union = np.logical_or(mask1_arr, mask2_arr)
        
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)
        
        if union_sum == 0:
            return 0.0
            
        return intersection_sum / union_sum
        
    except Exception as e:
        print(f"Error comparing masks: {e}")
        traceback.print_exc()
        return 0.0

def update_mask_tracking(current_masks: List[PhysicsMask]) -> None:
    """
    Update tracking statistics for PhysicsMask objects and handle rotations.
    """
    try:
        if current_masks is None:
            return
            
        frame_size = current_masks[0].mask.shape[:2]
        
        # Initialize tracking arrays if needed
        if len(state.mask_stability_counters) != len(current_masks):
            state.mask_stability_counters = [0] * len(current_masks)
        if len(state.mask_flags) != len(current_masks):
            state.mask_flags = ['dynamic'] * len(current_masks)
            
        # Initialize previous masks if this is the first frame
        if state.previous_masks is None:
            state.previous_masks = current_masks
            state.mask_change_scores = [0.0] * len(current_masks)
            return
            
        # Compare current masks with previous frame
        min_masks = min(len(current_masks), len(state.previous_masks))
        new_change_scores = []
        
        current_time = time.time()
        dt = current_time - state.last_update_time if state.last_update_time else 0.016
        state.last_update_time = current_time
        
        for i in range(min_masks):
            # Update physics including rotation
            current_masks[i].update(dt)
            
            similarity = compare_masks(
                current_masks[i],
                state.previous_masks[i]
            )
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
                
        # Transfer motion properties from previous masks
        for i in range(min_masks):
            if new_change_scores[i] > state.STABILITY_THRESHOLD:
                current_masks[i].dx = state.previous_masks[i].dx
                current_masks[i].dy = state.previous_masks[i].dy
                current_masks[i].angular_velocity = state.previous_masks[i].angular_velocity
                current_masks[i].angular_momentum = state.previous_masks[i].angular_momentum
        
        # Update tracking state
        state.mask_change_scores = new_change_scores
        state.current_masks = current_masks
        state.previous_masks = [mask.copy() for mask in current_masks]  # Deep copies
        
    except Exception as e:
        print(f"Error updating mask tracking: {e}")
        traceback.print_exc()

def create_physics_mask(mask_array: np.ndarray,
                       frame_size: Tuple[int, int],
                       mass: float = 1.0,
                       friction: float = 0.35,
                       pixel_content: Optional[np.ndarray] = None) -> PhysicsMask:
    """
    Create a new PhysicsMask object with pixel content and physics properties.
    
    Args:
        mask_array: Binary mask data
        frame_size: (height, width) of the frame
        mass: Physics mass property
        friction: Physics friction property
        pixel_content: Original pixel content from frame
        
    Returns:
        PhysicsMask: New physics-enabled mask object
    """
    # Find center of mass
    y_indices, x_indices = np.where(mask_array)
    if len(y_indices) == 0 or len(x_indices) == 0:
        center = (frame_size[1] // 2, frame_size[0] // 2)
    else:
        # Calculate actual center of mass
        center = (int(np.mean(x_indices)), int(np.mean(y_indices)))

    # Debug print for mask creation
    print(f"\nMask Creation Debug:")
    print(f"Mask shape: {mask_array.shape}")
    print(f"Mask pixels: {np.sum(mask_array)}")
    if pixel_content is not None:
        print(f"Pixel content shape: {pixel_content.shape}")
        print(f"Non-zero pixels in content: {np.count_nonzero(pixel_content)}")
    print(f"Center of mass: {center}")
    
    # Calculate moment of inertia
    if len(x_indices) > 0:
        dx = x_indices - center[0]
        dy = y_indices - center[1]
        r_squared = dx * dx + dy * dy
        moment_of_inertia = np.sum(r_squared) * mass / len(x_indices)
    else:
        moment_of_inertia = 1.0
    
    # Set initial lifetime based on size
    size_ratio = np.sum(mask_array) / (frame_size[0] * frame_size[1])
    lifetime = max(50.0, min(200.0, 100.0 * size_ratio))

    # Verify pixel content matches mask
    if pixel_content is not None:
        # Make sure pixel content is correctly masked
        pixel_content = pixel_content.copy()  # Make a copy to ensure we don't modify original
        pixel_content[~mask_array] = 0  # Zero out pixels outside the mask
        
        # Debug print for pixel content
        print(f"Verified pixel content non-zero pixels: {np.count_nonzero(pixel_content)}")
        print(f"Should match mask pixels: {np.sum(mask_array)}")
    
    return PhysicsMask(
        mask=mask_array,
        position=center,
        dx=0.0,
        dy=0.0,
        friction=friction,
        mass=mass,
        is_active=True,
        moment_of_inertia=moment_of_inertia,
        lifetime=lifetime,
        decay_rate=1.0,
        opacity=1.0,
        pixel_content=pixel_content
    )

def translate_and_rotate_mask(mask: PhysicsMask, 
                            dx: int, 
                            dy: int,
                            frame_size: Tuple[int, int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Translate and rotate a mask and its pixel content.
    
    Args:
        mask: PhysicsMask to transform
        dx: Horizontal translation
        dy: Vertical translation
        frame_size: (height, width) of frame
        
    Returns:
        Tuple of (translated/rotated mask, translated/rotated pixel content)
    """
    height, width = frame_size
    
    if abs(dx) > width or abs(dy) > height:
        return np.zeros_like(mask.mask, dtype=bool), None
    
    # Print debug info about transformation
    print(f"\nTransform Debug:")
    print(f"dx, dy: {dx}, {dy}")
    print(f"rotation: {mask.rotation}")
    if mask.pixel_content is not None:
        print(f"Pixel content stats before transform:")
        print(f"  Shape: {mask.pixel_content.shape}")
        print(f"  Non-zero pixels: {np.count_nonzero(mask.pixel_content)}")
    
    # Create transformation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Add rotation if needed
    if mask.rotation != 0:
        center = mask.center
        R = cv2.getRotationMatrix2D(center, mask.rotation, 1.0)
        M = np.matmul(R, np.vstack([M, [0, 0, 1]]))[:2]  # Combine transformations
    
    # Transform mask
    translated_mask = cv2.warpAffine(
        mask.mask.astype(np.uint8),
        M,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(bool)
    
    # Transform pixel content if available
    translated_content = None
    if mask.pixel_content is not None:
        # Ensure pixel content is preserved during transformation
        translated_content = cv2.warpAffine(
            mask.pixel_content,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Debug transformed content
        print(f"\nPixel content stats after transform:")
        print(f"  Shape: {translated_content.shape}")
        print(f"  Non-zero pixels: {np.count_nonzero(translated_content)}")
        
        # Apply mask to ensure no pixels leak outside
        translated_content[~translated_mask] = 0
    
    return translated_mask, translated_content

def calculate_collision_torque(mask1: PhysicsMask, 
                             mask2: PhysicsMask, 
                             collision_point: Tuple[float, float]) -> float:
    """
    Calculate torque from a collision between two masks.
    
    Args:
        mask1: First colliding mask
        mask2: Second colliding mask
        collision_point: Point of collision
        
    Returns:
        float: Resulting torque
    """
    # Vector from center of mass to collision point
    r1x = collision_point[0] - mask1.center[0]
    r1y = collision_point[1] - mask1.center[1]
    
    # Relative velocity at collision point
    rel_vx = mask2.dx - mask1.dx
    rel_vy = mask2.dy - mask1.dy
    
    # Cross product to get torque (r × F)
    # Force is approximated by relative velocity
    torque = (r1x * rel_vy - r1y * rel_vx) * mask2.mass
    
    return torque

def update_mask_content(mask: PhysicsMask, 
                       current_frame: np.ndarray,
                       force_update: bool = False) -> None:
    """
    Update the pixel content of a mask from the current frame.
    
    Args:
        mask: PhysicsMask to update
        current_frame: Current video frame
        force_update: Whether to force update even for moving masks
    """
    if not mask.is_active:
        return
        
    # Only update content for stationary masks unless forced
    if force_update or (abs(mask.dx) < 0.1 and abs(mask.dy) < 0.1 
                       and abs(mask.angular_velocity) < 0.1):
        # Create a copy of the frame section under the mask
        new_content = current_frame.copy()
        new_content[~mask.mask] = 0
        mask.pixel_content = new_content