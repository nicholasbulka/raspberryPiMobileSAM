import cv2
import numpy as np
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from scipy import ndimage

from .effects_parameters import (
    EffectParameters, MaskAnimator, EFFECTS_LIST,
    MIN_PIXELATION, MAX_PIXELATION, DEFAULT_PIXELATION
)

logger = logging.getLogger(__name__)

class EffectsManager:
    """
    Manages visual effects processing and application.
    Handles effect selection, parameter management, and frame processing.
    """
    
    def __init__(self):
        """Initialize effects manager with default settings"""
        self.current_effect = "none"
        self.effects_params = EffectParameters({})
        self.mask_animator = MaskAnimator()
        
        # Effect state tracking
        self.last_glitch_time = time.time()
        self.glitch_duration = 0.0
        self.current_glitch_seed = None
        self.wave_offset = 0.0
        self.last_wave_update = time.time()
        self.melt_buffer = None
        self.melt_positions = None

        # Performance tracking
        self.effect_timings = {
            'glitch': [],
            'wave': [],
            'melt': []
        }
        self.max_timing_samples = 100

        # Thread safety
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize effects system"""
        try:
            # Create test data for validation
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_mask = np.ones((100, 100), dtype=bool)
            
            # Test each effect type
            test_effects = ['none', 'blur', 'pixelate', 'outline', 'glitch', 'wave', 'melt']
            for effect in test_effects:
                self.current_effect = effect
                result = self.process_frame(test_frame, [test_mask])
                if result is None:
                    logger.error(f"Effect {effect} failed validation")
                    return False
            
            self.current_effect = 'none'  # Reset to default
            logger.info("Effects manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing effects: {e}")
            return False

    def get_effects_list(self) -> List[Dict[str, Any]]:
        """Get list of available effects with metadata"""
        return EFFECTS_LIST

    def set_effect(self, effect_name: str) -> bool:
        """Set current effect with validation"""
        with self._lock:
            if effect_name in [effect['name'] for effect in EFFECTS_LIST]:
                self.current_effect = effect_name
                # Reset effect-specific state
                self.last_glitch_time = time.time()
                self.glitch_duration = 0.0
                self.current_glitch_seed = None
                self.wave_offset = 0.0
                self.last_wave_update = time.time()
                self.melt_buffer = None
                self.melt_positions = None
                logger.info(f"Effect set to: {effect_name}")
                return True
            logger.warning(f"Attempted to set unknown effect: {effect_name}")
            return False

    def update_params(self, params: EffectParameters) -> None:
        """Update effect parameters"""
        with self._lock:
            self.effects_params = params

    def process_frame(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply current effect to frame with masks"""
        start_time = time.time()
        
        try:
            # Update mask animator with metadata
            if masks is not None and len(masks) > 0:
                metadata = [{"score": 1.0} for _ in masks]  # Default metadata
                self.mask_animator.update_masks([masks[0]], [metadata[0]])  # Only track first mask
                
                # Apply selected effect
                result = frame.copy()
                if self.current_effect == 'none':
                    result = self._effect_none(frame, masks)
                elif self.current_effect == 'blur':
                    result = self._effect_blur(frame, masks)
                elif self.current_effect == 'pixelate':
                    result = self._effect_pixelate(frame, masks)
                elif self.current_effect == 'outline':
                    result = self._effect_outline(frame, masks)
                elif self.current_effect == 'glitch':
                    result = self._effect_glitch(frame, masks)
                elif self.current_effect == 'wave':
                    result = self._effect_wave(frame, masks)
                elif self.current_effect == 'melt':
                    result = self._effect_melt(frame, masks)
                    
                # Update performance metrics
                process_time = time.time() - start_time
                self._update_timing(self.current_effect, process_time)
                
                return result
            
            return frame.copy()
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame.copy()

    def _update_timing(self, effect: str, duration: float) -> None:
        """Update performance metrics for effect"""
        if effect in self.effect_timings:
            self.effect_timings[effect].append(duration)
            if len(self.effect_timings[effect]) > self.max_timing_samples:
                self.effect_timings[effect].pop(0)

    def _effect_none(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Pass-through effect with optional mask highlighting"""
        result = frame.copy()
        for mask in masks:
            if mask is not None and mask.any():  # Check if mask has any True values
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                # Add subtle highlight to show mask
                result[mask_resized] = cv2.addWeighted(
                    result[mask_resized], 1.1, 
                    np.zeros_like(result[mask_resized]), 0, 0
                )
        return result

    def _effect_blur(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply blur effect to masked regions"""
        result = frame.copy()
        kernel_size = self.effects_params.get_param('strength', 2) * 2 + 1
        
        for mask in masks:
            if mask is not None and mask.any():  # Check if mask has any True values
                # Resize mask to match frame dimensions
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                
                # Apply blur
                blurred = cv2.GaussianBlur(
                    frame,
                    (kernel_size, kernel_size),
                    0
                )
                result[mask_resized] = blurred[mask_resized]
                
        return result

    def _effect_pixelate(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply pixelation effect to masked regions"""
        result = frame.copy()
        pixel_size = self.effects_params.get_param('pixelation', 6)
        
        for mask in masks:
            if mask is not None and mask.any():  # Check if mask has any True values
                # Resize mask to match frame dimensions
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                
                # Create pixelation effect
                h, w = frame.shape[:2]
                temp = cv2.resize(
                    frame,
                    (w // pixel_size, h // pixel_size),
                    interpolation=cv2.INTER_LINEAR
                )
                pixelated = cv2.resize(
                    temp,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Apply to masked region
                result[mask_resized] = pixelated[mask_resized]
                
        return result

    def _resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Safely resize mask to target dimensions"""
        if mask.shape != target_shape:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            return resized > 0  # Convert back to boolean
        return mask

    def _effect_outline(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply outline effect to masked regions"""
        result = frame.copy()
        thickness = self.effects_params.get_param('strength', 2)
        
        for mask in masks:
            if mask is not None and mask.any():  # Check if mask has any True values
                # Get mask contours
                mask_uint8 = cv2.resize(
                    mask.astype(np.uint8) * 255,
                    (frame.shape[1], frame.shape[0])
                )
                contours, _ = cv2.findContours(
                    mask_uint8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Draw contours
                cv2.drawContours(
                    result,
                    contours,
                    -1,
                    (0, 255, 0),
                    thickness
                )
                
        return result

    def _effect_glitch(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply glitch effect to masked regions"""
        result = frame.copy()
        current_time = time.time()
        intensity = self.effects_params.get_param('intensity', 5) / 10.0
        
        for mask in masks:
            if mask is not None and mask.any():
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                
                # Check if we need new glitch parameters
                if current_time - self.last_glitch_time > self.glitch_duration:
                    self.current_glitch_seed = np.random.randint(0, 1000000)
                    self.glitch_duration = 0.1 + (intensity * 0.3)
                    self.last_glitch_time = current_time
                
                # Use consistent seed for current glitch duration
                np.random.seed(self.current_glitch_seed)
                
                if np.random.random() < intensity:
                    # Color channel shift
                    channel = np.random.randint(0, 3)
                    shift = np.random.randint(-10, 11)
                    masked_data = result[mask_resized].copy()
                    masked_data[..., channel] = np.roll(masked_data[..., channel], shift)
                    result[mask_resized] = masked_data
                
                if np.random.random() < intensity:
                    # Vertical slice offset
                    slice_height = np.random.randint(10, 30)
                    y_start = np.random.randint(0, frame.shape[0] - slice_height)
                    x_offset = np.random.randint(-20, 21)
                    
                    if x_offset != 0:
                        slice_mask = mask_resized[y_start:y_start + slice_height]
                        result_slice = result[y_start:y_start + slice_height]
                        result[y_start:y_start + slice_height][slice_mask] = np.roll(
                            result_slice[slice_mask],
                            x_offset,
                            axis=0
                        )
        
        return result

    def _effect_wave(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply wave distortion effect to masked regions"""
        result = frame.copy()
        current_time = time.time()
        
        # Update wave offset based on time and speed
        delta_time = current_time - self.last_wave_update
        self.wave_offset += delta_time * self.effects_params.get_param('speed', 5)
        self.last_wave_update = current_time
        
        for mask in masks:
            if mask is not None and mask.any():
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                
                # Create displacement maps
                y_coords, x_coords = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
                
                # Calculate wave displacement
                amplitude = self.effects_params.get_param('amplitude', 10)
                frequency = self.effects_params.get_param('frequency', 3)
                
                displacement = amplitude * np.sin(
                    2 * np.pi * frequency * y_coords / frame.shape[0] + self.wave_offset
                )
                
                # Apply displacement where mask is True
                for y in range(frame.shape[0]):
                    mask_row = mask_resized[y]
                    if mask_row.any():
                        x_shift = int(displacement[y])
                        if x_shift != 0:
                            row_data = result[y][mask_row].copy()
                            result[y][mask_row] = np.roll(row_data, x_shift, axis=0)
        
        return result

    def _effect_melt(self, frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Apply melting effect to masked regions"""
        result = frame.copy()
        
        # Initialize or reset melt buffer if needed
        if (self.melt_buffer is None or 
            self.melt_buffer.shape != frame.shape or 
            self.melt_positions is None):
            self.melt_buffer = frame.copy()
            self.melt_positions = np.zeros(frame.shape[1], dtype=np.int32)
        
        strength = self.effects_params.get_param('strength', 2)
        speed = self.effects_params.get_param('speed', 5)
        
        for mask in masks:
            if mask is not None and mask.any():
                mask_resized = self._resize_mask(mask, frame.shape[:2])
                
                # Update melt positions
                for x in range(frame.shape[1]):
                    if mask_resized[:, x].any():
                        # Calculate flow rate based on brightness and parameters
                        brightness = np.mean(frame[mask_resized[:, x], :]) / 255.0
                        flow_rate = int(strength * speed * (0.5 + brightness))
                        
                        # Update position with bounds checking
                        self.melt_positions[x] = min(
                            self.melt_positions[x] + flow_rate,
                            frame.shape[0] - 1
                        )
                        
                        # Apply melting effect
                        if self.melt_positions[x] > 0:
                            y_start = max(0, self.melt_positions[x] - flow_rate)
                            y_end = self.melt_positions[x]
                            
                            # Stretch pixels vertically
                            mask_column = mask_resized[y_start:y_end, x]
                            if mask_column.any():
                                result[y_start:y_end, x] = frame[y_start, x]
        
        # Apply slight blur for smoother effect
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        metrics = {}
        for effect, timings in self.effect_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                metrics[effect] = {
                    'avg_process_time': avg_time,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0
                }
        
        return {
            'current_effect': self.current_effect,
            'params': self.effects_params.get_all_params(),
            'performance': {
                'effect_timings': metrics,
                'current_frame_time': time.time() - self.last_wave_update
            },
            'debug': {
                'active_mask': self.mask_animator.get_mask_info(),
                'effects': {
                    'glitch': {
                        'last_update': self.last_glitch_time,
                        'duration': self.glitch_duration,
                        'current_seed': self.current_glitch_seed
                    },
                    'wave': {
                        'offset': self.wave_offset,
                        'last_update': self.last_wave_update
                    },
                    'melt': {
                        'has_buffer': self.melt_buffer is not None,
                        'buffer_shape': self.melt_buffer.shape if self.melt_buffer is not None else None,
                        'has_positions': self.melt_positions is not None
                    }
                }
            }
        }

    def formatJSON(self, data, level=0):
        """Format JSON data for debug display"""
        indent = '  ' * level
        
        if isinstance(data, (list, tuple)):
            if not data:
                return '[]'
            items = []
            for item in data:
                formatted_item = self.formatJSON(item, level + 1)
                items.append(indent + "  " + formatted_item)
            return "[\n" + ",\n".join(items) + "\n" + indent + "]"
        
        if data is None:
            return 'null'
        
        if not isinstance(data, dict):
            return str(data)
        
        if not data:
            return '{}'
        
        items = []
        for key, value in data.items():
            formatted_value = self.formatJSON(value, level + 1)
            items.append(f'{indent}  "{key}": {formatted_value}')
        
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"

    def filterMasks(self, searchTerm: str) -> None:
        """Filter masks in debug display"""
        self.mask_animator.filter_masks(searchTerm)

    def toggleAllMasks(self, expand: bool) -> None:
        """Expand or collapse all masks in debug display"""
        self.mask_animator.toggle_all_masks(expand)

    def cleanup(self) -> None:
        """Clean up resources"""
        self.melt_buffer = None
        self.melt_positions = None