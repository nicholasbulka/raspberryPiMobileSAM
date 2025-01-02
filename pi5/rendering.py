# rendering.py
import cv2
import numpy as np
import time
import colorsys
import traceback
from typing import Optional, List, Tuple
import shared_state as state

# Visualization parameters
ALPHA_BASE = 0.25        # Base opacity for mask overlays
ALPHA_INCREMENT = 0.15   # Additional opacity per layer
MOTION_THRESHOLD = 0.1   # Minimum motion intensity to trigger reveal
MOTION_SMOOTHING = 0.8   # Temporal smoothing factor

# Cache for generated colors
mask_colors = {}

# Debug settings
debug_enabled = True
last_debug_time = time.time()
debug_interval = 1.0  # Seconds between debug prints

def debug_array(name: str, arr: np.ndarray) -> None:
    """Print debug information about numpy arrays."""
    global last_debug_time
    
    if not debug_enabled:
        return
        
    current_time = time.time()
    if current_time - last_debug_time < debug_interval:
        return
        
    if arr is None:
        print(f"{name} is None")
        return
        
    print(f"{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {np.min(arr)}, max: {np.max(arr)}")
    print(f"  unique values: {np.unique(arr)[:5]}...")
    
    last_debug_time = current_time

def generate_color(index: int) -> np.ndarray:
    """Generate a vibrant, visually distinct color for mask visualization."""
    if index in mask_colors:
        return mask_colors[index]
        
    golden_ratio = 0.618033988749895
    hue = (index * golden_ratio) % 1.0
    saturation = np.random.uniform(0.7, 1.0)
    value = np.random.uniform(0.8, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    variation = np.random.uniform(-20, 20, 3)
    rgb = np.clip(rgb + variation, 0, 255)
    
    color = rgb.astype(np.uint8)
    mask_colors[index] = color
    return color

def create_mask_visualization(frame: np.ndarray,
                            masks: Optional[List[np.ndarray]] = None,
                            mask_flags: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a visualization of the masks overlaid on the frame.
    The visualization shows each mask with a distinct color and opacity.
    """
    if masks is None or len(masks) == 0:
        return frame.copy()
        
    result = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize mask flags if not provided
    if mask_flags is None:
        mask_flags = ['dynamic'] * len(masks)
        
    # Create mask overlays
    for i, (mask, flag) in enumerate(zip(masks, mask_flags)):
        # Ensure mask is properly sized
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        color = [255, 255, 255] if flag == 'stable' else generate_color(i)
        overlay[mask_resized] = color
        
        # Calculate alpha for this layer
        alpha = ALPHA_BASE + (ALPHA_INCREMENT * (i / len(masks)))
        
        # Blend overlay with result
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        # Draw contours
        mask_uint8 = mask_resized.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        contour_color = [0, 0, 0] if flag == 'stable' else color.tolist()
        cv2.drawContours(result, contours, -1, contour_color, 1)
    
    return result

def apply_motion_reveal(base_frame: np.ndarray,
                       original_frame: np.ndarray,
                       motion_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply motion-based reveal effect. Where motion is detected,
    show the original frame instead of the visualization.
    """
    if motion_mask is None:
        return base_frame
        
    # Ensure proper types and ranges
    base_frame = base_frame.astype(np.float32)
    original_frame = original_frame.astype(np.float32)
    motion_mask = motion_mask.astype(np.float32)
    
    # Clip motion mask to valid range
    motion_mask = np.clip(motion_mask, 0, 1)
    
    # Create 3-channel motion mask
    motion_mask_3ch = np.dstack([motion_mask] * 3)
    
    # Debug information
    if debug_enabled:
        debug_array("base_frame", base_frame)
        debug_array("original_frame", original_frame)
        debug_array("motion_mask", motion_mask)
        debug_array("motion_mask_3ch", motion_mask_3ch)
    
    # Where motion is detected (motion_mask > threshold), 
    # reveal the original frame instead of the visualization
    motion_significant = motion_mask_3ch > MOTION_THRESHOLD
    result = np.where(motion_significant, original_frame, base_frame)
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def render_loop():
    """
    Main rendering loop that runs as a separate thread.
    Handles all visualization and frame composition tasks.
    """
    print("Starting render loop")
    last_render_time = time.time()
    target_interval = 1/60  # Target 60 FPS for rendering
    
    try:
        while True:
            try:
                current_time = time.time()
                time_since_last = current_time - last_render_time
                
                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue
                
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                        
                    frame = state.current_frame.copy()
                    masks = state.current_masks
                    motion_mask = state.motion_mask
                
                # Create visualization with masks
                visualization = create_mask_visualization(
                    frame, 
                    masks,
                    state.mask_flags if masks is not None else None
                )
                
                # Apply motion reveal effect
                if motion_mask is not None:
                    result = apply_motion_reveal(visualization, frame, motion_mask)
                else:
                    result = visualization
                
                # Update the rendered frame in shared state
                with state.frame_lock:
                    state.rendered_frame = result
                
                last_render_time = current_time
                
            except Exception as e:
                print(f"Error in render loop: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Fatal error in render loop: {str(e)}")
        traceback.print_exc()