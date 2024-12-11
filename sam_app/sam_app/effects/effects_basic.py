from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
import time
from effects_base import (
    EffectParameters, safe_array_access,
    MIN_PIXELATION, MAX_PIXELATION, DEFAULT_PIXELATION
)

class BasicEffects:
    """
    Implements fundamental video effect operations that serve as building blocks
    for more complex effects. Focuses on core transformations like pixelation
    and color manipulation.
    """
    
    def __init__(self):
        # Track performance metrics for optimization
        self.last_process_time = 0.0
        self.frame_count = 0
        self.total_process_time = 0.0
        
        # Cache for optimizing repeated operations
        self._pixelation_cache = {}
        self._cache_max_size = 100
        
    def _get_cache_key(self, shape: Tuple[int, ...], pixelation: int) -> str:
        """Generate a unique key for caching pixelation calculations"""
        return f"{shape}_{pixelation}"
    
    def _clear_cache(self) -> None:
        """Clear the pixelation cache when it gets too large"""
        if len(self._pixelation_cache) > self._cache_max_size:
            self._pixelation_cache.clear()

    def apply_pixelation(self, frame: np.ndarray, pixelation_factor: int) -> np.ndarray:
        """
        Apply pixelation effect with caching and bounds checking.
        Args:
            frame: Input video frame
            pixelation_factor: Level of pixelation (higher = more pixelated)
        Returns:
            Pixelated frame
        """
        try:
            # Input validation
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
            
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                raise ValueError("Frame has zero dimensions")
            
            # Bound pixelation factor
            pixelation_factor = max(MIN_PIXELATION, min(int(pixelation_factor), MAX_PIXELATION))
            
            # Check cache for dimension calculations
            cache_key = self._get_cache_key(frame.shape, pixelation_factor)
            if cache_key in self._pixelation_cache:
                small_h, small_w = self._pixelation_cache[cache_key]
            else:
                # Calculate new dimensions ensuring they're never zero
                small_h = max(1, h // pixelation_factor)
                small_w = max(1, w // pixelation_factor)
                self._pixelation_cache[cache_key] = (small_h, small_w)
                self._clear_cache()

            # Apply pixelation using optimal interpolation methods
            small = cv2.resize(frame, (small_w, small_h), 
                             interpolation=cv2.INTER_LINEAR)
            
            # Use INTER_NEAREST for upscaling to maintain pixelated look
            return cv2.resize(small, (w, h), 
                            interpolation=cv2.INTER_NEAREST)
            
        except Exception as e:
            print(f"Error in pixelation: {e}")
            return frame

    def adjust_color_levels(self, frame: np.ndarray, 
                          brightness: float = 1.0,
                          contrast: float = 1.0,
                          gamma: float = 1.0) -> np.ndarray:
        """
        Adjust color levels of the frame with gamma correction.
        Args:
            frame: Input frame
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)
            gamma: Gamma correction value (1.0 = no change)
        Returns:
            Color-adjusted frame
        """
        try:
            # Input validation
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
            
            # Normalize parameters
            brightness = max(0.0, min(brightness, 3.0))
            contrast = max(0.0, min(contrast, 3.0))
            gamma = max(0.1, min(gamma, 2.5))
            
            # Convert to float32 for calculations
            adjusted = frame.astype(np.float32) / 255.0
            
            # Apply gamma correction
            if gamma != 1.0:
                adjusted = np.power(adjusted, gamma)
            
            # Apply contrast
            if contrast != 1.0:
                adjusted = (adjusted - 0.5) * contrast + 0.5
            
            # Apply brightness
            if brightness != 1.0:
                adjusted = adjusted * brightness
            
            # Clip values and convert back to uint8
            return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Error in color adjustment: {e}")
            return frame

    def apply_color_quantization(self, frame: np.ndarray, levels: int = 4) -> np.ndarray:
        """
        Reduce the number of colors in the frame for retro effects.
        Args:
            frame: Input frame
            levels: Number of levels per channel (e.g., 4 = 64 total colors)
        Returns:
            Color-quantized frame
        """
        try:
            # Input validation
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
                
            # Bound levels to reasonable range
            levels = max(2, min(int(levels), 32))
            
            # Calculate quantization step
            step = 255 // (levels - 1)
            
            # Quantize colors
            quantized = frame // step * step
            
            return quantized.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in color quantization: {e}")
            return frame

    def apply_noise(self, frame: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Add controlled random noise to the frame.
        Args:
            frame: Input frame
            intensity: Noise intensity (0.0 to 1.0)
        Returns:
            Frame with added noise
        """
        try:
            # Input validation
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
                
            # Bound intensity
            intensity = max(0.0, min(float(intensity), 1.0))
            
            # Generate noise matching frame dimensions
            noise = np.random.normal(0, intensity * 255, frame.shape)
            
            # Add noise and clip values
            noisy = frame + noise.astype(frame.dtype)
            return np.clip(noisy, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Error applying noise: {e}")
            return frame

    def get_performance_metrics(self) -> Dict[str, float]:
        """Return current performance metrics"""
        if self.frame_count == 0:
            return {'avg_process_time': 0, 'fps': 0}
            
        avg_time = self.total_process_time / self.frame_count
        return {
            'avg_process_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }

    def process_frame(self, frame: np.ndarray, params: EffectParameters) -> np.ndarray:
        """
        Main processing function that applies basic effects based on parameters.
        Args:
            frame: Input video frame
            params: Effect parameters including pixelation, color adjustments, etc.
        Returns:
            Processed frame with basic effects applied
        """
        start_time = time.time()
        
        try:
            result = frame.copy()
            
            # Apply pixelation if specified
            if params.get_param('pixelation', 0) > 1:
                result = self.apply_pixelation(result, params.pixelation)
            
            # Apply color quantization for 8-bit effect if enabled
            if params.get_param('8bit', False):
                result = self.apply_color_quantization(result, 4)
            
            # Apply any color adjustments
            brightness = params.get_param('brightness', 1.0)
            contrast = params.get_param('contrast', 1.0)
            gamma = params.get_param('gamma', 1.0)
            
            if any(x != 1.0 for x in (brightness, contrast, gamma)):
                result = self.adjust_color_levels(
                    result, brightness, contrast, gamma
                )
            
            # Add noise if specified
            noise_intensity = params.get_param('noise', 0.0)
            if noise_intensity > 0:
                result = self.apply_noise(result, noise_intensity)
            
            # Update performance metrics
            process_time = time.time() - start_time
            self.last_process_time = process_time
            self.total_process_time += process_time
            self.frame_count += 1
            
            return result
            
        except Exception as e:
            print(f"Error in basic effects processing: {e}")
            return frame

# Global instance for basic effects
basic_effects = BasicEffects()
