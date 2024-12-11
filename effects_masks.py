from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
import time
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from effects_base import (
    EffectParameters, safe_array_access, MaskAnimator,
    MIN_PIXELATION, MAX_PIXELATION
)
from effects_basic import basic_effects

class MaskEffects:
    """
    Implements effects that interact with segmentation masks.
    Handles growing, shrinking, and other mask-based transformations
    while managing smooth transitions and animations.
    """
    
    def __init__(self):
        # Initialize mask animator for handling mask states
        self.mask_animator = MaskAnimator()
        
        # Transition states for smooth effect changes
        self.transition_state = {
            'active_masks': set(),
            'fade_out_masks': {},  # mask_id -> fade progress
            'fade_in_masks': {},   # mask_id -> fade progress
            'transition_start': 0.0
        }
        
        # Effect-specific states
        self.pulse_states = {}  # mask_id -> pulse parameters
        self.glow_buffers = {}  # mask_id -> glow buffer
        
        # Performance monitoring
        self.timing_stats = {
            'mask_processing': [],
            'effect_application': [],
            'transition_handling': []
        }
        self.max_timing_samples = 100

    def _update_timing(self, category: str, duration: float) -> None:
        """Track performance metrics for different processing stages"""
        self.timing_stats[category].append(duration)
        if len(self.timing_stats[category]) > self.max_timing_samples:
            self.timing_stats[category].pop(0)

    def _prepare_mask(self, mask: np.ndarray, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Prepare a mask for effect application by ensuring correct dimensions
        and handling edge cases.
        """
        try:
            start_time = time.time()
            
            if mask is None:
                return np.zeros(frame_shape[:2], dtype=bool)
            
            # Resize mask if dimensions don't match
            if mask.shape != frame_shape[:2]:
                resized_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame_shape[1], frame_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                mask = resized_mask > 0
            
            # Clean up mask edges
            mask = binary_erosion(mask, iterations=1)
            mask = binary_dilation(mask, iterations=1)
            
            self._update_timing('mask_processing', time.time() - start_time)
            return mask
            
        except Exception as e:
            print(f"Error preparing mask: {e}")
            return np.zeros(frame_shape[:2], dtype=bool)

    def _create_glow(self, mask: np.ndarray, intensity: float = 1.0, 
                    color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Create a glowing effect around mask edges.
        """
        try:
            # Create distance field from mask
            dist = cv2.distanceTransform(
                mask.astype(np.uint8), 
                cv2.DIST_L2, 
                5
            )
            
            # Normalize and create glow falloff
            glow = np.exp(-dist / (10 * intensity))
            
            # Create colored glow
            glow_colored = np.zeros((*mask.shape, 3), dtype=np.float32)
            for i in range(3):
                glow_colored[..., i] = glow * (color[i] / 255.0)
            
            return glow_colored
            
        except Exception as e:
            print(f"Error creating glow: {e}")
            return np.zeros((*mask.shape, 3), dtype=np.float32)

    def apply_grow_effect(self, frame: np.ndarray, masks: List[np.ndarray], 
                         params: EffectParameters) -> np.ndarray:
        """
        Apply growing effect to masks with smooth transitions and glow effects.
        """
        try:
            start_time = time.time()
            
            # Apply basic pixelation first
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            
            if not masks:
                return result
            
            # Update mask animator with current masks
            self.mask_animator.switch_interval = params.get_param('switch_interval', 2.0)
            self.mask_animator.update_masks(masks, [{'score': 1.0} for _ in masks])
            
            # Get current animation state
            current_state = self.mask_animator.get_current_state()
            
            if current_state['active_mask'] is not None:
                mask_index = current_state['active_mask']
                current_mask = self._prepare_mask(masks[mask_index], frame.shape)
                
                # Calculate growth based on animation phase
                if current_state['phase'] == 'growing':
                    growth_factor = current_state['size']
                else:
                    growth_factor = max(1, current_state['size'])
                
                # Apply growth with smooth edges
                grown_mask = binary_dilation(current_mask, iterations=growth_factor)
                smooth_mask = gaussian_filter(grown_mask.astype(float), sigma=1)
                
                # Create glow effect
                glow = self._create_glow(
                    grown_mask,
                    intensity=params.get_param('glow_intensity', 1.0),
                    color=(64, 200, 255)  # Blue-ish glow
                )
                
                # Apply mask effect with glow
                mask_region = smooth_mask[..., np.newaxis]
                result = result * (1 - mask_region) + \
                        (result * 1.2 + glow * 255) * mask_region
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            self._update_timing('effect_application', time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"Error in grow effect: {e}")
            return frame

    def apply_shrink_effect(self, frame: np.ndarray, masks: List[np.ndarray], 
                           params: EffectParameters) -> np.ndarray:
        """
        Apply shrinking effect to masks with edge preservation and dark borders.
        """
        try:
            start_time = time.time()
            
            # Apply basic pixelation first
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            
            if not masks:
                return result
            
            current_time = time.time()
            speed = params.get_param('speed', 5)
            strength = params.get_param('strength', 2)
            
            # Calculate shrink factor with time-based animation
            time_factor = (np.sin(current_time * speed) + 1) / 2
            shrink_factor = int(strength * 2 * time_factor) + 1
            
            # Process each mask
            for mask in masks:
                prepared_mask = self._prepare_mask(mask, frame.shape)
                
                # Create multiple shrink levels for smooth transition
                shrink_levels = []
                alphas = []
                
                for i in range(shrink_factor + 1):
                    shrunk = binary_erosion(prepared_mask, iterations=i)
                    shrink_levels.append(shrunk)
                    alphas.append(1.0 - (i / shrink_factor))
                
                # Apply multi-level shrink effect
                for shrunk, alpha in zip(shrink_levels, alphas):
                    # Create dark border
                    border = binary_dilation(shrunk, iterations=1) ^ shrunk
                    result[border] = np.clip(result[border] * 0.5, 0, 255)
                    
                    # Darken non-mask regions
                    result[~shrunk] = np.clip(
                        result[~shrunk] * (0.8 + 0.2 * alpha), 
                        0, 
                        255
                    ).astype(np.uint8)
            
            self._update_timing('effect_application', time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"Error in shrink effect: {e}")
            return frame

    def apply_pulse_effect(self, frame: np.ndarray, masks: List[np.ndarray],
                          params: EffectParameters) -> np.ndarray:
        """
        Apply pulsing effect to masks with color variation and smooth transitions.
        """
        try:
            start_time = time.time()
            
            result = basic_effects.apply_pixelation(frame, params.pixelation)
            
            if not masks:
                return result
            
            current_time = time.time()
            
            # Process each mask
            for i, mask in enumerate(masks):
                if i not in self.pulse_states:
                    self.pulse_states[i] = {
                        'phase': 0.0,
                        'frequency': np.random.uniform(0.5, 2.0),
                        'color_shift': np.random.uniform(0, 2 * np.pi, 3)
                    }
                
                # Update pulse phase
                state = self.pulse_states[i]
                state['phase'] += current_time * state['frequency']
                
                # Calculate pulse intensity
                intensity = 0.3 + 0.7 * (np.sin(state['phase']) + 1) / 2
                
                # Create color variation
                color = np.array([
                    np.sin(state['phase'] + state['color_shift'][0]),
                    np.sin(state['phase'] + state['color_shift'][1]),
                    np.sin(state['phase'] + state['color_shift'][2])
                ])
                color = (color + 1) / 2  # Normalize to 0-1
                
                # Apply effect
                prepared_mask = self._prepare_mask(mask, frame.shape)
                glow = self._create_glow(
                    prepared_mask,
                    intensity=intensity,
                    color=tuple((color * 255).astype(int))
                )
                
                mask_region = prepared_mask[..., np.newaxis]
                result = result * (1 - mask_region) + \
                        (result * (1 + 0.2 * intensity) + glow * 255) * mask_region
                
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            self._update_timing('effect_application', time.time() - start_time)
            return result
            
        except Exception as e:
            print(f"Error in pulse effect: {e}")
            return frame

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return detailed performance metrics for all processing stages"""
        metrics = {}
        for category, timings in self.timing_stats.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                metrics[category] = {
                    'avg_process_time': avg_time,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0,
                    'min_time': min(timings),
                    'max_time': max(timings)
                }
            else:
                metrics[category] = {
                    'avg_process_time': 0,
                    'fps': 0,
                    'min_time': 0,
                    'max_time': 0
                }
        return metrics

    def reset_state(self) -> None:
        """Reset all internal states and buffers"""
        self.mask_animator = MaskAnimator()
        self.transition_state = {
            'active_masks': set(),
            'fade_out_masks': {},
            'fade_in_masks': {},
            'transition_start': 0.0
        }
        self.pulse_states.clear()
        self.glow_buffers.clear()

# Global instance for mask effects
mask_effects = MaskEffects()
