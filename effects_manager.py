from typing import List, Dict, Any, Optional
import numpy as np
import time
import json
from effects_base import EffectParameters, EFFECTS_LIST
from effects_basic import basic_effects
from effects_advanced import advanced_effects
from effects_masks import mask_effects

class EffectsManager:
    """
    Central manager for all effect operations. Coordinates between different
    effect modules and provides a unified interface for effect application.
    """
    
    def __init__(self):
        self.current_effect = 'none'
        self.current_params = {}
        self.last_effect_time = time.time()
        self.frame_count = 0
        
        # Debug and performance monitoring
        self.debug_info = {
            'effect_history': [],
            'performance_metrics': {},
            'current_state': {}
        }
        self.max_history_entries = 100

    def update_params(self, params: Dict[str, Any]) -> None:
        """Update effect parameters with validation"""
        self.current_params = EffectParameters(params)
        self._update_debug_info('params_update', params)

    def set_effect(self, effect_name: str) -> None:
        """Change current effect with proper state cleanup"""
        if effect_name != self.current_effect:
            # Clean up previous effect state
            advanced_effects.reset_state()
            mask_effects.reset_state()
            
            self.current_effect = effect_name
            self._update_debug_info('effect_change', {'new_effect': effect_name})

    def process_frame(self, frame: np.ndarray, 
                     masks: Optional[List[np.ndarray]] = None,
                     mask_metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Process a frame with the current effect and parameters.
        Handles all effect types and maintains performance metrics.
        """
        start_time = time.time()
        
        try:
            # Ensure we have valid input
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
            
            params = self.current_params
            result = frame.copy()
            
            # Apply appropriate effect based on type
            if self.current_effect == 'none':
                result = basic_effects.process_frame(result, params)
            
            elif self.current_effect in ['glitch', 'wave', 'melt']:
                if self.current_effect == 'glitch':
                    result = advanced_effects.apply_glitch(result, params)
                elif self.current_effect == 'wave':
                    result = advanced_effects.apply_wave(result, params)
                elif self.current_effect == 'melt':
                    result = advanced_effects.apply_melt(result, params)
                    
            elif self.current_effect in ['grow', 'shrink'] and masks is not None:
                if self.current_effect == 'grow':
                    result = mask_effects.apply_grow_effect(result, masks, params)
                elif self.current_effect == 'shrink':
                    result = mask_effects.apply_shrink_effect(result, masks, params)
            
            # Apply any global post-processing
            if params.get_param('8bit', False):
                result = basic_effects.apply_color_quantization(result, 4)
            
            # Update performance metrics
            self._update_performance_metrics(time.time() - start_time)
            self.frame_count += 1
            
            return result
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

    def _update_performance_metrics(self, process_time: float) -> None:
        """Update and maintain performance metrics"""
        metrics = {
            'basic': basic_effects.get_performance_metrics(),
            'advanced': advanced_effects.get_performance_metrics(),
            'masks': mask_effects.get_performance_metrics(),
            'total_time': process_time,
            'fps': 1.0 / process_time if process_time > 0 else 0,
            'frame_count': self.frame_count
        }
        
        self.debug_info['performance_metrics'] = metrics

    def _update_debug_info(self, event_type: str, data: Dict[str, Any]) -> None:
        """Update debug information with event history"""
        entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data
        }
        
        self.debug_info['effect_history'].append(entry)
        if len(self.debug_info['effect_history']) > self.max_history_entries:
            self.debug_info['effect_history'].pop(0)
        
        self.debug_info['current_state'] = {
            'effect': self.current_effect,
            'params': self.current_params._original_params,
            'frame_count': self.frame_count
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get current debug information including performance metrics"""
        return self.debug_info

    @staticmethod
    def get_effects_list() -> List[Dict[str, Any]]:
        """Return the list of available effects and their parameters"""
        return EFFECTS_LIST

# Global effects manager instance
effects_manager = EffectsManager()
