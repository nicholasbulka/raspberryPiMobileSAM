from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import time
from effects_base import (
    EffectParameters, safe_array_access,
    MIN_PIXELATION, MAX_PIXELATION, DEFAULT_PIXELATION
)
from effects_basic import basic_effects

class AdvancedEffects:
    """
    Implements complex video effects building upon basic effects.
    Handles advanced transformations like glitch, wave, and melt effects.
    """
    
    def __init__(self):
        # Maintain state for time-based effects
        self.last_glitch_time = time.time()
        self.glitch_duration = 0.0
        self.current_glitch_seed = None
        
        # Wave effect state
        self.wave_offset = 0.0
        self.last_wave_update = time.time()
        
        # Melt effect state
        self.melt_buffer = None
        self.melt_positions = None
        
        # Performance tracking
        self.effect_timings = {
            'glitch': [],
            'wave': [],
            'melt': []
        }
        self.max_timing_samples = 100

    def _update_timing(self, effect: str, duration: float) -> None:
        """Track performance metrics for each effect"""
        self.effect_timings[effect].append(duration)
        if len(self.effect_timings[effect]) > self.max_timing_samples:
            self.effect_timings[effect].pop(0)

    def apply_glitch(self, frame: np.ndarray, params: EffectParameters) -> np.ndarray:
        """
        Apply glitch effect with controlled randomization and temporal coherence.
        Uses a seed system to maintain consistent glitches over small time windows.
        """
        try:
            start_time = time.time()
            
            # Always apply basic pixelation first
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            height, width = result.shape[:2]
            
            current_time = time.time()
            intensity = params.intensity / 10.0  # Normalize to 0-1 range
            
            # Check if we need a new glitch seed
            if (current_time - self.last_glitch_time > self.glitch_duration or 
                self.current_glitch_seed is None):
                
                # Generate a new seed within uint32 range
                self.current_glitch_seed = int((current_time * 1000) % (2**32 - 1))
                self.glitch_duration = 0.1 + (intensity * 0.3)  # 0.1-0.4 seconds
                self.last_glitch_time = current_time
            
            # Use the current seed for consistent glitching
            np.random.seed(self.current_glitch_seed)
            
            # Calculate number of glitch effects based on intensity
            num_glitches = int(intensity * 10)
            
            # Apply various glitch effects
            for _ in range(num_glitches):
                # Random slice parameters
                slice_height = np.random.randint(5, int(height * 0.2))
                y_start = np.random.randint(0, height - slice_height)
                x_offset = np.random.randint(-int(width * 0.1), int(width * 0.1))
                
                # Apply slice offset
                if x_offset != 0:
                    slice_data = result[y_start:y_start + slice_height].copy()
                    if x_offset > 0:
                        result[y_start:y_start + slice_height, x_offset:] = \
                            safe_array_access(slice_data, (slice(None), slice(None, -x_offset)))
                    else:
                        result[y_start:y_start + slice_height, :x_offset] = \
                            safe_array_access(slice_data, (slice(None), slice(-x_offset, None)))
                
                # Random color channel shift
                if np.random.random() < intensity:
                    channel = np.random.randint(0, 3)
                    shift = np.random.randint(-20, 21)
                    result[y_start:y_start + slice_height, :, channel] = \
                        np.roll(result[y_start:y_start + slice_height, :, channel], shift, axis=1)
                
                # Occasional color distortion
                if np.random.random() < intensity * 0.5:
                    color_scale = np.random.uniform(0.8, 1.2, (1, 1, 3))
                    result[y_start:y_start + slice_height] = \
                        np.clip(result[y_start:y_start + slice_height] * color_scale, 0, 255)
            
            # Add scan lines effect based on intensity
            if intensity > 0.5:
                scan_lines = np.sin(np.linspace(0, 10 * np.pi, height)) * intensity * 0.3
                scan_lines = np.repeat(scan_lines[:, np.newaxis], width, axis=1)
                scan_lines = np.repeat(scan_lines[:, :, np.newaxis], 3, axis=2)
                result = np.clip(result * (1 + scan_lines), 0, 255).astype(np.uint8)
            
            self._update_timing('glitch', time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"Error in glitch effect: {e}")
            return frame

    def apply_wave(self, frame: np.ndarray, params: EffectParameters) -> np.ndarray:
        """
        Apply wave distortion effect with smooth animation and edge handling.
        """
        try:
            start_time = time.time()
            
            # Apply basic pixelation first
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            height, width = result.shape[:2]
            
            # Update wave offset based on time and speed
            current_time = time.time()
            delta_time = current_time - self.last_wave_update
            self.wave_offset += delta_time * params.speed
            self.last_wave_update = current_time
            
            # Create displacement maps
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            
            # Calculate wave displacement
            wave_x = params.amplitude * np.sin(
                2 * np.pi * params.frequency * y_coords / height + self.wave_offset
            )
            wave_y = params.amplitude * np.cos(
                2 * np.pi * params.frequency * x_coords / width + self.wave_offset
            )
            
            # Add secondary waves for more complex motion
            wave_x += params.amplitude * 0.5 * np.sin(
                4 * np.pi * params.frequency * y_coords / height - self.wave_offset * 0.5
            )
            wave_y += params.amplitude * 0.5 * np.cos(
                4 * np.pi * params.frequency * x_coords / width - self.wave_offset * 0.5
            )
            
            # Create sampling coordinates
            sample_x = (x_coords + wave_x).astype(np.float32)
            sample_y = (y_coords + wave_y).astype(np.float32)
            
            # Bound the coordinates to prevent edge artifacts
            sample_x = np.clip(sample_x, 0, width - 1)
            sample_y = np.clip(sample_y, 0, height - 1)
            
            # Remap the image using the wave displacement
            mapped = cv2.remap(
                result,
                sample_x,
                sample_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            self._update_timing('wave', time.time() - start_time)
            return mapped
            
        except Exception as e:
            print(f"Error in wave effect: {e}")
            return frame

    def apply_melt(self, frame: np.ndarray, params: EffectParameters) -> np.ndarray:
        """
        Apply melting effect with dynamic flow and edge preservation.
        """
        try:
            start_time = time.time()
            
            # Apply basic pixelation first
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            height, width = result.shape[:2]
            
            # Initialize or reset melt buffer and positions
            if (self.melt_buffer is None or 
                self.melt_buffer.shape[:2] != (height, width)):
                self.melt_buffer = result.copy()
                self.melt_positions = np.zeros(width, dtype=np.int32)
            
            # Update melt positions based on image intensity
            for x in range(width):
                if self.melt_positions[x] < height - 1:
                    # Calculate flow rate based on pixel intensity and parameters
                    intensity = np.mean(result[self.melt_positions[x], x]) / 255.0
                    flow_rate = int(params.strength * (1.0 + intensity))
                    
                    # Update position with bounds checking
                    new_pos = min(self.melt_positions[x] + flow_rate, height - 1)
                    
                    # Copy pixels down
                    if new_pos > self.melt_positions[x]:
                        self.melt_buffer[self.melt_positions[x]:new_pos, x] = \
                            result[self.melt_positions[x], x]
                    
                    self.melt_positions[x] = new_pos
            
            # Apply smooth transitions between melted regions
            kernel_size = 3
            self.melt_buffer = cv2.GaussianBlur(
                self.melt_buffer, (kernel_size, kernel_size), 0
            )
            
            # Blend with original image for smoother effect
            blend_factor = np.linspace(1.0, 0.5, height)[:, np.newaxis, np.newaxis]
            self.melt_buffer = (self.melt_buffer * blend_factor + 
                              result * (1 - blend_factor))
            
            self._update_timing('melt', time.time() - start_time)
            return self.melt_buffer.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in melt effect: {e}")
            return frame

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return performance metrics for all effects"""
        metrics = {}
        for effect, timings in self.effect_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                metrics[effect] = {
                    'avg_process_time': avg_time,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0,
                    'min_time': min(timings),
                    'max_time': max(timings)
                }
            else:
                metrics[effect] = {
                    'avg_process_time': 0,
                    'fps': 0,
                    'min_time': 0,
                    'max_time': 0
                }
        return metrics

    def reset_state(self) -> None:
        """Reset all internal state (useful when switching effects)"""
        self.last_glitch_time = time.time()
        self.glitch_duration = 0.0
        self.current_glitch_seed = None
        self.wave_offset = 0.0
        self.last_wave_update = time.time()
        self.melt_buffer = None
        self.melt_positions = None

# Global instance for advanced effects
advanced_effects = AdvancedEffects()
