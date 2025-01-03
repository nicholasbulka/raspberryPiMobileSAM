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
MOTION_THRESHOLD = 0.02  # Threshold for motion to trigger mask removal
MOTION_SMOOTHING = 0.8   # Temporal smoothing factor

# Cache for generated colors to maintain consistent mask coloring
mask_colors = {}

# Debug settings for array visualization
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
    rgb = np.clip(rgb + variation, 20, 155)
    
    color = rgb.astype(np.uint8)
    mask_colors[index] = color
    return color

def create_mask_visualization(frame: np.ndarray,
                            masks: Optional[List[np.ndarray]] = None,
                            mask_flags: Optional[List[str]] = None,
                            motion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a visualization of the masks overlaid on the frame.
    Now includes motion-based mask removal.
    
    Args:
        frame: Base frame to overlay masks on
        masks: List of segmentation masks
        mask_flags: List of mask states ('stable' or 'dynamic')
        motion_mask: Optional motion mask to control mask visibility
    """
    if masks is None or len(masks) == 0:
        return frame.copy()
        
    result = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize mask flags if not provided
    if mask_flags is None:
        mask_flags = ['dynamic'] * len(masks)
        
    # If we have a motion mask, prepare it for blending
    if motion_mask is not None:
        # Ensure motion mask is in the right format
        motion_mask = motion_mask.astype(np.float32)
        motion_mask = cv2.resize(motion_mask, (frame_width, frame_height))
        motion_mask = np.clip(motion_mask, 0, 1)
        # Create inverse mask (1 where no motion, 0 where motion)
        inverse_motion = 1.0 - motion_mask
        inverse_motion_3ch = np.dstack([inverse_motion] * 3)
    
    # Create mask overlays
    for i, (mask, flag) in enumerate(zip(masks, mask_flags)):
        # Ensure mask is properly sized
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Create colored overlay
        overlay = np.zeros_like(frame, dtype=np.float32)
        color = [255, 255, 255] if flag == 'stable' else generate_color(i)
        overlay[mask_resized] = color
        
        # Calculate alpha for this layer
        alpha = ALPHA_BASE + (ALPHA_INCREMENT * (i / len(masks)))
        
        if motion_mask is not None:
            # Apply motion-based transparency
            # Where motion is detected (motion_mask > threshold):
            #   - The inverse_motion_3ch will be close to 0, making the mask disappear
            # Where no motion is detected:
            #   - The inverse_motion_3ch will be close to 1, showing the mask normally
            motion_significant = motion_mask > MOTION_THRESHOLD
            alpha = alpha * inverse_motion_3ch
        
        # Convert to float32 for blending
        result_float = result.astype(np.float32)
        
        # Blend overlay with result using the motion-adjusted alpha
        result = cv2.addWeighted(
            result_float,
            1.0,
            overlay,
            alpha,
            0
        ).astype(np.uint8)
        
        # Draw contours where there isn't significant motion
        mask_uint8 = mask_resized.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if motion_mask is not None:
            # Only draw contours where there isn't significant motion
            contour_color = [0, 0, 0] if flag == 'stable' else color.tolist()
            cv2.drawContours(result, contours, -1, contour_color, 1)
    
    return result

def render_loop():
    """
    Main rendering loop that runs as a separate thread.
    Handles visualization composition and motion-based mask removal.
    """
    print("Starting render loop")
    last_render_time = time.time()
    target_interval = 1/60  # Target 60 FPS for rendering
    
    try:
        while True:
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - last_render_time
                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue
                
                # Get current state with thread safety
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                        
                    # Get all necessary components
                    frame = state.current_frame.copy()
                    debug_frame = state.raw_processed_frame
                    masks = state.current_masks
                    motion_mask = state.motion_mask
                
                # Use debug frame as base if available
                base_frame = debug_frame if debug_frame is not None else frame
                
                # Create visualization with masks and motion-based removal
                result = create_mask_visualization(
                    base_frame, 
                    masks,
                    state.mask_flags if masks is not None else None,
                    motion_mask
                )
                
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
