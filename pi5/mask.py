import cv2
import numpy as np
import traceback
from typing import List, Optional, Tuple
import shared_state as state
from colors import color_scheme, generate_color
from mask_types import PhysicsMask
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
    Update tracking statistics for PhysicsMask objects.
    Ensure we use .mask when doing numpy operations on the masks.
    """
    try:
        if current_masks is None:
            return
            
        frame_size = current_masks[0].mask.shape[:2]  # Access shape from the numpy array
        
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
        
        for i in range(min_masks):
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
                
        # Transfer velocity from previous masks if they match well
        for i in range(min_masks):
            if new_change_scores[i] > state.STABILITY_THRESHOLD:
                current_masks[i].dx = state.previous_masks[i].dx
                current_masks[i].dy = state.previous_masks[i].dy
                
        # Update tracking state
        state.mask_change_scores = new_change_scores
        state.current_masks = current_masks
        state.previous_masks = [mask.copy() for mask in current_masks]  # Create deep copies
        
    except Exception as e:
        print(f"Error updating mask tracking: {e}")
        traceback.print_exc()

def create_physics_mask(mask_array: np.ndarray,
                       frame_size: Tuple[int, int],
                       mass: float = 1.0,
                       friction: float = 0.95) -> PhysicsMask:
    """
    Create a new PhysicsMask object from a binary mask array.
    
    Args:
        mask_array: Binary mask data
        frame_size: (height, width) of the frame
        mass: Physics mass property (affects motion response)
        friction: Physics friction property (affects velocity decay)
        
    Returns:
        PhysicsMask: New physics-enabled mask object
    """
    # Find center of the mask
    y_indices, x_indices = np.where(mask_array)
    if len(y_indices) == 0 or len(x_indices) == 0:
        center = (frame_size[1] // 2, frame_size[0] // 2)
    else:
        center = (int(np.mean(x_indices)), int(np.mean(y_indices)))
    
    return PhysicsMask(
        mask=mask_array,
        position=center,
        dx=0.0,
        dy=0.0,
        friction=friction,
        mass=mass,
        is_active=True
    )

def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a mask while preserving its binary nature using nearest neighbor interpolation.
    
    Args:
        mask: Source binary mask array
        target_size: (width, height) tuple
        
    Returns:
        Resized binary mask array
    """
    return cv2.resize(
        mask.astype(np.uint8),
        target_size,
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

def get_mask_visualization(frame: np.ndarray,
                         physics_mask: PhysicsMask,
                         color: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """
    Create a colored visualization of a physics-enabled mask.
    
    Args:
        frame: Original frame
        physics_mask: PhysicsMask object to visualize
        color: BGR color array for the mask
        alpha: Opacity of the overlay
        
    Returns:
        Frame with mask visualization overlay
    """
    # Create colored overlay
    overlay = np.zeros_like(frame)
    overlay[physics_mask.mask] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    # Draw contours for better edge visibility
    mask_uint8 = physics_mask.mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw contours and velocity vectors
    cv2.drawContours(result, contours, -1, color.tolist(), 1)
    
    # Draw velocity vector if mask is moving
    if abs(physics_mask.dx) > 0.1 or abs(physics_mask.dy) > 0.1:
        center = physics_mask.center
        end_point = (
            int(center[0] + physics_mask.dx * 10),
            int(center[1] + physics_mask.dy * 10)
        )
        cv2.arrowedLine(result, center, end_point, color.tolist(), 2)
    
    return result