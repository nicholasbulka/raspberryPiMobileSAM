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

def create_mask_visualization(frame: np.ndarray,
                            masks: Optional[List[PhysicsMask]] = None,
                            mask_flags: Optional[List[str]] = None,
                            motion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a visualization of the masks overlaid on the frame with proper alpha handling.
    Now properly handles PhysicsMask objects by accessing their underlying numpy arrays.
    """
    if masks is None or len(masks) == 0:
        return frame.copy()
        
    # Start with a fresh copy of the frame converted to float32
    result = frame.astype(np.float32) / 255.0
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize accumulated alpha channel
    accumulated_alpha = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Initialize mask flags if not provided
    if mask_flags is None:
        mask_flags = ['dynamic'] * len(masks)
        
    # Process motion mask if provided
    if motion_mask is not None:
        motion_mask = cv2.resize(motion_mask.astype(np.float32), 
                               (frame_width, frame_height))
        motion_mask = np.clip(motion_mask, 0, 1)
        inverse_motion = 1.0 - motion_mask
    
    # Create mask overlays with premultiplied alpha
    for i, (physics_mask, flag) in enumerate(zip(masks, mask_flags)):
        try:
            # Access the underlying numpy array from the PhysicsMask object
            mask_data = physics_mask.mask
            
            # Ensure mask is properly sized
            mask_resized = cv2.resize(
                mask_data.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Calculate this layer's alpha
            base_alpha = ALPHA_BASE + (ALPHA_INCREMENT * (i / len(masks)))
            
            # Adjust alpha based on motion if needed
            if motion_mask is not None:
                layer_alpha = base_alpha * inverse_motion
            else:
                layer_alpha = np.full((frame_height, frame_width), base_alpha, 
                                    dtype=np.float32)
                
            # Only apply alpha where mask exists
            layer_alpha = layer_alpha * mask_resized
            
            # Create color overlay
            color = np.array([1.0, 1.0, 1.0] if flag == 'stable' else 
                            generate_color(i).astype(np.float32) / 255.0)
            
            # Create premultiplied color overlay
            overlay = np.zeros((frame_height, frame_width, 3), dtype=np.float32)
            for c in range(3):
                overlay[:, :, c] = color[c] * layer_alpha
                
            # Composite using premultiplied alpha
            available_alpha = 1.0 - accumulated_alpha
            layer_contribution = layer_alpha * available_alpha
            accumulated_alpha += layer_contribution
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1.0 - layer_contribution) + \
                                 overlay[:, :, c]
                                 
        except Exception as e:
            print(f"Error processing mask {i}: {str(e)}")
            continue
    
    # Ensure final result is properly bounded and converted back to uint8
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    # Draw contours
    for i, (physics_mask, flag) in enumerate(zip(masks, mask_flags)):
        try:
            # Access the underlying numpy array for contour detection
            mask_data = physics_mask.mask
            mask_uint8 = cv2.resize(
                mask_data.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST
            ) * 255
            
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Only draw contours where motion is minimal
            contour_color = [0, 0, 0] if flag == 'stable' else \
                           generate_color(i).tolist()
            cv2.drawContours(result, contours, -1, contour_color, 1)
            
        except Exception as e:
            print(f"Error drawing contours for mask {i}: {str(e)}")
            continue
    
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