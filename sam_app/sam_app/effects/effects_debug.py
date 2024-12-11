import numpy as np
import threading
import time
from typing import Dict, Any, List

class EffectsDebugManager:
    def __init__(self):
        # Debug information
        self.debug_info = {
            'active_mask': None,
            'mask_history': [],
            'performance': {
                'effect_time': 0.0,
                'effect_count': 0,
                'mask_count': 0
            }
        }
        
        # Thread safety
        self._debug_lock = threading.Lock()

    def get_debug_info(self, current_effect: str, effects_params: Any, effects_list: List) -> Dict[str, Any]:
        """
        Get comprehensive debug information with proper type conversion.
        """
        with self._debug_lock:
            debug_data = {
                'current_effect': {
                    'name': current_effect,
                    'params': self._convert_numpy_types(effects_params)
                },
                'effects': effects_list,
                'metrics': self._convert_numpy_types(self.debug_info['performance']),
                'debug': {
                    'active_mask': self._convert_numpy_types(self.debug_info['active_mask']),
                    'mask_history': [self._convert_numpy_types(m) for m in self.debug_info['mask_history']],
                    'performance': self._convert_numpy_types(self.debug_info['performance'])
                }
            }
            return debug_data

    def update_metrics(self, process_time: float, mask_count: int) -> None:
        """Update performance metrics thread-safely"""
        with self._debug_lock:
            self.debug_info['performance']['effect_time'] = process_time
            self.debug_info['performance']['effect_count'] += 1
            self.debug_info['performance']['mask_count'] = mask_count

    def update_debug_info(self, masks: List[np.ndarray]) -> None:
        """Update debug information thread-safely with type conversion"""
        with self._debug_lock:
            if len(masks) > 0:
                # Update active mask info
                mask = masks[0]
                active_mask = {
                    'size': [int(x) for x in mask.shape],
                    'area': int(np.sum(mask)),
                    'bounds': self._get_mask_bounds(mask)
                }
                self.debug_info['active_mask'] = active_mask
                
                # Update mask history
                self.debug_info['mask_history'].append({
                    'timestamp': time.time(),
                    'count': len(masks),
                    'sizes': [[int(x) for x in m.shape] for m in masks]
                })
                
                # Limit history length
                if len(self.debug_info['mask_history']) > 100:
                    self.debug_info['mask_history'].pop(0)

    def _get_mask_bounds(self, mask: np.ndarray) -> Dict[str, int]:
        """Calculate bounding box for mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return {'top': int(rmin), 'bottom': int(rmax), 
                'left': int(cmin), 'right': int(cmax)}

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj