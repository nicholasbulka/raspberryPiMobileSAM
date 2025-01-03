import cv2
import numpy as np
import time
import traceback
from typing import Optional, List, Tuple
import shared_state as state
from colors import generate_color
from mask import resize_mask
from mask_types import PhysicsMask

# Visualization parameters
ALPHA_BASE = 0.25        # Base opacity for mask overlays
ALPHA_INCREMENT = 0.15   # Additional opacity per layer
MOTION_THRESHOLD = 0.02  # Threshold for motion to trigger mask removal
MOTION_SMOOTHING = 0.8   # Temporal smoothing factor

# Cache for generated colors to maintain consistent mask coloring
mask_colors = {}

def debug_array(name: str, arr: np.ndarray) -> None:
    """Print debug information about numpy arrays."""
    if arr is None:
        print(f"{name} is None")
        return
        
    print(f"{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {np.min(arr)}, max: {np.max(arr)}")
    print(f"  unique values: {np.unique(arr)[:5]}...")

def create_mask_visualization(frame, masks, mask_flags=None, motion_mask=None):
    """
    Create a visualization of the masks with improved error handling and debugging.
    Each mask is rendered as a colored overlay on the frame.
    """
    print(f"\n=== Visualization Debug ===")
    print(f"Frame shape: {frame.shape}")
    print(f"Number of masks: {len(masks) if masks else 0}")
    
    if masks is None or len(masks) == 0:
        print("No masks to visualize")
        return frame.copy()
        
    # Start with a fresh copy of the frame
    result = frame.astype(np.float32) / 255.0
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize accumulated alpha channel
    accumulated_alpha = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Ensure we have flags for each mask
    if mask_flags is None or len(mask_flags) != len(masks):
        mask_flags = ['dynamic'] * len(masks)
    
    print(f"Processing {len(masks)} masks with flags: {mask_flags}")
    
    # Create mask overlays with premultiplied alpha
    for i, physics_mask in enumerate(masks):
        try:
            if not hasattr(physics_mask, 'mask'):
                print(f"Mask {i} is not a valid PhysicsMask object")
                continue

            # Access the underlying numpy array from the PhysicsMask object
            mask_data = physics_mask.mask
            print(f"Mask {i}: shape={mask_data.shape}, sum={np.sum(mask_data)}")
            
            if not isinstance(mask_data, np.ndarray):
                print(f"Mask {i} data is not a numpy array")
                continue
                
            if mask_data.shape[:2] != (frame_height, frame_width):
                print(f"Resizing mask from {mask_data.shape} to {(frame_height, frame_width)}")
                mask_resized = cv2.resize(
                    mask_data.astype(np.uint8),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask_data.astype(bool)
            
            # Calculate per-mask alpha and color
            base_alpha = 0.4  # Make masks more visible
            flag = mask_flags[i] if i < len(mask_flags) else 'dynamic'
            
            # Create distinct colors for each mask
            color = generate_color(i).astype(np.float32) / 255.0
            
            # Create the alpha channel for this mask
            layer_alpha = np.zeros((frame_height, frame_width), dtype=np.float32)
            layer_alpha[mask_resized] = base_alpha
            
            # Create colored overlay
            for c in range(3):
                result[mask_resized, c] = (
                    result[mask_resized, c] * (1 - base_alpha) +
                    color[c] * base_alpha
                )
            
            # Draw contours around the mask
            mask_uint8 = mask_resized.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert back to uint8 for contour drawing
            result_uint8 = (result * 255).astype(np.uint8)
            cv2.drawContours(
                result_uint8,
                contours,
                -1,
                color.tolist(),
                2  # Thicker contours for visibility
            )
            result = result_uint8.astype(np.float32) / 255.0
            
            print(f"Successfully processed mask {i}")
            
        except Exception as e:
            print(f"Error processing mask {i}: {str(e)}")
            traceback.print_exc()
            continue
    
    # Convert back to uint8 for display
    result = (result * 255).astype(np.uint8)
    print("Visualization complete")
    
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