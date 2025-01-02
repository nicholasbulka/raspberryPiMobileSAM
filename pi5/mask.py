import cv2
import numpy as np
import traceback
from typing import List, Optional, Tuple
import shared_state as state
from colors import color_scheme, generate_color

def compare_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compare two masks using Intersection over Union (IoU). This function
    helps us track how masks change between frames by measuring their overlap.
    A score of 1.0 means identical masks, while 0.0 means no overlap.
    
    Args:
        mask1: First mask array
        mask2: Second mask array
        
    Returns:
        Similarity score between 0 and 1
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    try:
        # Convert masks to boolean arrays for logical operations
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        # Calculate IoU score
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)
        
        if union_sum == 0:
            return 0.0
            
        return intersection_sum / union_sum
        
    except Exception as e:
        print(f"Error comparing masks: {e}")
        traceback.print_exc()
        return 0.0

def update_mask_tracking(current_masks: List[np.ndarray]) -> None:
    """
    Update tracking statistics and stability flags for masks. This function
    helps us identify which masks are stable (not changing much between frames)
    and which are dynamic, allowing us to apply different visual effects.
    
    Args:
        current_masks: List of current frame mask arrays
    """
    try:
        if current_masks is None:
            return
            
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
            # Calculate similarity with previous frame
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

def create_mask_visualization(frame: np.ndarray,
                            mask: np.ndarray,
                            color: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
    """
    Create a colored visualization overlay for a mask. This function
    helps us see the detected masks by coloring them distinctly.
    
    Args:
        frame: Original frame
        mask: Boolean mask array
        color: BGR color array for the mask
        alpha: Opacity of the overlay
        
    Returns:
        Frame with mask overlay
    """
    # Create colored overlay
    overlay = np.zeros_like(frame)
    overlay[mask] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    # Draw contours for better edge visibility
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    cv2.drawContours(result, contours, -1, color.tolist(), 1)
    
    return result

def resize_mask(mask: np.ndarray,
                target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a mask to target dimensions while preserving its binary nature.
    Uses nearest neighbor interpolation to avoid creating partial mask values.
    
    Args:
        mask: Source mask array
        target_size: (width, height) tuple
        
    Returns:
        Resized binary mask
    """
    return cv2.resize(
        mask.astype(np.uint8),
        target_size,
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)