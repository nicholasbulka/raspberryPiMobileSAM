import cv2
import numpy as np
import time
import traceback
from typing import Optional, List, Tuple
import shared_state as state
from colors import generate_color
from mask import translate_and_rotate_mask, update_mask_content
from mask_types import PhysicsMask

# Visualization parameters
MASK_OPACITY = 0.4           # Base opacity for masks
MOTION_VECTOR_SCALE = 2.0    # Scale factor for motion vectors
VECTOR_MIN_MAGNITUDE = 0.5   # Minimum magnitude to draw vectors
MASK_BORDER_THICKNESS = 2    # Thickness of mask borders
MOTION_TRAIL_LENGTH = 5      # Number of positions to keep for trails
DEBUG_MODE = False           # Enable additional debug visualization

class MaskVisualizer:
    """Handles advanced visualization of physics-enabled masks."""
    
    def __init__(self):
        self.mask_trails = {}  # Store recent positions for motion trails
        self.last_update = time.time()
        self.debug_info = {}   # Store debug information per mask
    
    def update_trails(self, mask_id: int, position: Tuple[int, int]):
        """Update motion trails for a mask."""
        if mask_id not in self.mask_trails:
            self.mask_trails[mask_id] = []
        
        self.mask_trails[mask_id].append(position)
        if len(self.mask_trails[mask_id]) > MOTION_TRAIL_LENGTH:
            self.mask_trails[mask_id].pop(0)
    
    def draw_motion_trails(self, frame: np.ndarray, mask_id: int, color: np.ndarray):
        """Draw motion trails for a mask."""
        if mask_id in self.mask_trails and len(self.mask_trails[mask_id]) > 1:
            points = np.array(self.mask_trails[mask_id], dtype=np.int32)
            cv2.polylines(
                frame,
                [points],
                False,
                color.tolist(),
                2,
                cv2.LINE_AA
            )
    
    def draw_physics_vectors(self, frame: np.ndarray, 
                           mask: PhysicsMask, 
                           color: np.ndarray):
        """Draw velocity and angular velocity vectors."""
        if not mask.is_active:
            return
            
        center = mask.center
        
        # Draw linear velocity vector
        if abs(mask.dx) > VECTOR_MIN_MAGNITUDE or abs(mask.dy) > VECTOR_MIN_MAGNITUDE:
            end_point = (
                int(center[0] + mask.dx * MOTION_VECTOR_SCALE),
                int(center[1] + mask.dy * MOTION_VECTOR_SCALE)
            )
            cv2.arrowedLine(
                frame,
                center,
                end_point,
                color.tolist(),
                2,
                cv2.LINE_AA,
                tipLength=0.3
            )
        
        # Draw angular velocity indicator
        if abs(mask.angular_velocity) > 1.0:  # Only draw if rotating significantly
            radius = 20  # Radius of rotation indicator
            start_angle = mask.rotation
            end_angle = start_angle + 30 * np.sign(mask.angular_velocity)
            
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0,
                start_angle,
                end_angle,
                color.tolist(),
                2,
                cv2.LINE_AA
            )
    
    def draw_debug_info(self, frame: np.ndarray, 
                       mask: PhysicsMask, 
                       mask_id: int,
                       color: np.ndarray):
        """Draw debug information for a mask."""
        if not DEBUG_MODE:
            return
            
        info_text = [
            f"ID: {mask_id}",
            f"Vel: ({mask.dx:.1f}, {mask.dy:.1f})",
            f"Ang: {mask.angular_velocity:.1f}°/s",
            f"KE: {mask.kinetic_energy:.1f}"
        ]
        
        # Position text above mask
        x1, y1, _, _ = mask.bounds
        y = max(20, y1 - 10)  # Ensure text stays in frame
        
        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (x1, y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                1,
                cv2.LINE_AA
            )

def create_mask_visualization(frame: np.ndarray,
                            masks: List[PhysicsMask],
                            mask_flags: Optional[List[str]] = None) -> np.ndarray:
    """Create enhanced visualization of physics-enabled masks."""
    if not masks:
        return frame.copy()
    
    # Initialize visualization
    result = frame.astype(np.float32) / 255.0
    height, width = frame.shape[:2]
    visualizer = MaskVisualizer()
    
    # Ensure we have flags for each mask
    if mask_flags is None or len(mask_flags) != len(masks):
        mask_flags = ['dynamic'] * len(masks)
    
    try:
        # Process each mask
        for i, physics_mask in enumerate(masks):
            if not physics_mask.is_active:
                continue
            
            # Generate consistent color for this mask
            color = generate_color(i).astype(np.float32) / 255.0
            
            # Update pixel content for relatively stationary masks
            if physics_mask.pixel_content is None:
                update_mask_content(physics_mask, frame)
            
            # Get transformed mask and content
            transformed_mask, transformed_content = translate_and_rotate_mask(
                physics_mask,
                int(physics_mask.dx),
                int(physics_mask.dy),
                (height, width)
            )
            
            if transformed_content is not None:
                # Blend pixel content with opacity
                mask_overlay = transformed_content.astype(np.float32) / 255.0
                opacity = physics_mask.opacity
                result = cv2.addWeighted(
                    result,
                    1.0,
                    mask_overlay,
                    opacity,
                    0
                )
            else:
                # Fallback to color overlay if no pixel content
                mask_overlay = np.zeros((height, width, 3), dtype=np.float32)
                mask_overlay[transformed_mask] = color
                opacity = physics_mask.opacity * MASK_OPACITY
                result = cv2.addWeighted(
                    result,
                    1.0,
                    mask_overlay,
                    opacity,
                    0
                )
            
            # Draw mask border
            mask_uint8 = transformed_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Scale border color by opacity
            border_color = tuple((c * physics_mask.opacity).item() for c in color * 255)
            
            cv2.drawContours(
                result,
                contours,
                -1,
                border_color,
                MASK_BORDER_THICKNESS
            )
            
            # Draw physics visualization
            visualizer.draw_physics_vectors(result, physics_mask, color * 255)
            
            # Update and draw motion trails
            visualizer.update_trails(i, physics_mask.center)
            visualizer.draw_motion_trails(result, i, color * 255)
            
            # Draw debug info if enabled
            visualizer.draw_debug_info(result, physics_mask, i, color * 255)
    
    except Exception as e:
        print(f"Error in mask visualization: {str(e)}")
        traceback.print_exc()
    
    # Convert back to uint8
    return (result * 255).astype(np.uint8)

def render_loop() -> None:
    """Main rendering loop for visualization composition."""
    print("Starting render loop")
    last_render_time = time.time()
    target_interval = 1/60  # Target 60 FPS
    
    try:
        while True:
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - last_render_time
                if time_since_last < target_interval:
                    time.sleep(target_interval - time_since_last)
                    continue
                
                # Get current state safely
                with state.frame_lock:
                    if state.current_frame is None:
                        continue
                        
                    frame = state.current_frame.copy()
                    debug_frame = state.raw_processed_frame
                    masks = state.current_masks
                    mask_flags = state.mask_flags if masks else None
                
                # Create visualization
                base_frame = debug_frame if debug_frame is not None else frame
                result = create_mask_visualization(
                    base_frame,
                    masks,
                    mask_flags
                )
                
                # Update rendered frame
                with state.frame_lock:
                    state.rendered_frame = result
                
                last_render_time = current_time
                
            except Exception as e:
                print(f"Render loop error: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Fatal render error: {str(e)}")
        traceback.print_exc()

def debug_array(name: str, arr: np.ndarray) -> None:
    """Print debug information about numpy arrays."""
    if arr is None:
        print(f"{name} is None")
        return
        
    print(f"{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {np.min(arr)}, max: {np.max(arr)}")
    if arr.size > 0:
        print(f"  mean: {np.mean(arr):.2f}")
        print(f"  unique values: {np.unique(arr)[:5]}...")