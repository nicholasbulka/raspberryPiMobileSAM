import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
import random
import time

# Core constants for effect parameters
MAX_PIXELATION = 20
MIN_PIXELATION = 1
DEFAULT_PIXELATION = 6

MAX_INTENSITY = 10
MIN_INTENSITY = 1
DEFAULT_INTENSITY = 5

MAX_SPEED = 10
MIN_SPEED = 1
DEFAULT_SPEED = 5

MAX_STRENGTH = 5
MIN_STRENGTH = 1
DEFAULT_STRENGTH = 2

MAX_AMPLITUDE = 20
MIN_AMPLITUDE = 1
DEFAULT_AMPLITUDE = 10

MAX_FREQUENCY = 10
MIN_FREQUENCY = 1
DEFAULT_FREQUENCY = 3

@dataclass
class MaskState:
    mask_index: int
    phase: str
    size: int
    wall_touches: int
    original_mask: np.ndarray
    start_time: float
    transition_duration: float

class EffectParameters:
    def __init__(self, params: Dict[str, Any]):
        self.pixelation = max(MIN_PIXELATION, min(params.get('pixelation', DEFAULT_PIXELATION), MAX_PIXELATION))
        self.speed = max(MIN_SPEED, min(params.get('speed', DEFAULT_SPEED), MAX_SPEED))
        self.strength = max(MIN_STRENGTH, min(params.get('strength', DEFAULT_STRENGTH), MAX_STRENGTH))
        self.intensity = max(MIN_INTENSITY, min(params.get('intensity', DEFAULT_INTENSITY), MAX_INTENSITY))
        self.amplitude = max(MIN_AMPLITUDE, min(params.get('amplitude', DEFAULT_AMPLITUDE), MAX_AMPLITUDE))
        self.frequency = max(MIN_FREQUENCY, min(params.get('frequency', DEFAULT_FREQUENCY), MAX_FREQUENCY))
        self.switch_interval = max(1, min(params.get('switch_interval', 2), 10))
        self._original_params = params

    def get_param(self, name: str, default: Any = None) -> Any:
        return self._original_params.get(name, default)

class MaskAnimator:
    def __init__(self):
        self.current_mask_state: Optional[MaskState] = None
        self.masks: List[np.ndarray] = []
        self.mask_metadata: List[Dict[str, Any]] = []
        self.last_switch_time = time.time()
        self.switch_interval = 2.0
        self.performance_metrics = {
            'fps': 0,
            'process_time': 0,
            'num_masks': 0
        }

    def update_masks(self, masks: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        self.masks = masks
        self.mask_metadata = metadata
        self.performance_metrics['num_masks'] = len(masks)
        
        if not self.current_mask_state and masks:
            self.select_new_mask()

    def select_new_mask(self) -> None:
        if not self.masks:
            self.current_mask_state = None
            return

        mask_index = random.randint(0, len(self.masks) - 1)
        self.current_mask_state = MaskState(
            mask_index=mask_index,
            phase='growing',
            size=1,
            wall_touches=0,
            original_mask=self.masks[mask_index].copy(),
            start_time=time.time(),
            transition_duration=random.uniform(0.5, 1.5)
        )
        self.last_switch_time = time.time()

    def check_wall_touches(self, mask: np.ndarray) -> int:
        h, w = mask.shape
        touches = 0
        if np.any(mask[0, :]):
            touches += 1
        if np.any(mask[-1, :]):
            touches += 1
        if np.any(mask[:, 0]):
            touches += 1
        if np.any(mask[:, -1]):
            touches += 1
        return touches

    def get_mask_info(self) -> Dict[str, Any]:
        if not self.current_mask_state:
            return {"active_mask": None}

        mask_meta = self.mask_metadata[self.current_mask_state.mask_index]
        return {
            "active_mask": {
                "index": self.current_mask_state.mask_index,
                "phase": self.current_mask_state.phase,
                "size": self.current_mask_state.size,
                "wall_touches": self.current_mask_state.wall_touches,
                "score": mask_meta.get("score", 0),
                "area": float(np.sum(self.current_mask_state.original_mask)),
                "time_in_state": time.time() - self.current_mask_state.start_time
            },
            "masks": [
                {
                    "index": i,
                    "score": meta.get("score", 0),
                    "area": float(np.sum(mask))
                }
                for i, (mask, meta) in enumerate(zip(self.masks, self.mask_metadata))
            ],
            "performance": self.performance_metrics
        }

    def safe_resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask.shape != target_shape:
            return cv2.resize(
                mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ) > 0
        return mask

    def apply_mask_effect(self, frame: np.ndarray, mask: np.ndarray, effect_type: str = 'highlight') -> np.ndarray:
        result = frame.copy()
        if effect_type == 'highlight':
            result[mask] = np.clip(frame[mask] * 1.2, 0, 255).astype(np.uint8)
        elif effect_type == 'darken':
            result[~mask] = np.clip(frame[~mask] * 0.8, 0, 255).astype(np.uint8)
        return result

    def process_frame(self, frame: np.ndarray, params: EffectParameters) -> np.ndarray:
        start_time = time.time()
        
        if not self.current_mask_state or not self.masks:
            return frame

        result = frame.copy()
        current_time = time.time()

        if current_time - self.last_switch_time > params.switch_interval:
            if self.current_mask_state.phase == 'shrinking' and self.current_mask_state.size <= 1:
                self.select_new_mask()
                return frame

        mask = self.safe_resize_mask(self.current_mask_state.original_mask, frame.shape[:2])

        if self.current_mask_state.phase == 'growing':
            grown_mask = self.grow_mask(mask, self.current_mask_state.size)
            wall_touches = self.check_wall_touches(grown_mask)

            if wall_touches >= 3:
                self.current_mask_state.phase = 'shrinking'
            else:
                self.current_mask_state.size += 1
                self.current_mask_state.wall_touches = wall_touches
                result = self.apply_mask_effect(frame, grown_mask, 'highlight')
        else:
            if self.current_mask_state.size > 1:
                self.current_mask_state.size -= 1
                grown_mask = self.grow_mask(mask, self.current_mask_state.size)
                result = self.apply_mask_effect(frame, grown_mask, 'highlight')

        process_time = time.time() - start_time
        self.update_performance_metrics(process_time)
        
        return result

    def grow_mask(self, mask: np.ndarray, size: int) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=size)

    def update_performance_metrics(self, process_time: float) -> None:
        self.performance_metrics['process_time'] = process_time
        if process_time > 0:
            self.performance_metrics['fps'] = 1.0 / process_time

def safe_array_access(arr: np.ndarray, indices: Union[slice, Tuple[slice, ...]], default_value: Any = 0) -> np.ndarray:
    try:
        return arr[indices]
    except IndexError:
        if isinstance(indices, tuple):
            shape = tuple(idx.stop - idx.start for idx in indices if isinstance(idx, slice))
            return np.full(shape, default_value, dtype=arr.dtype)
        return default_value

EFFECTS_LIST = [
    {
        'name': 'none',
        'label': 'No Effect',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'}
        ]
    },
    {
        'name': 'melt',
        'label': 'Melt',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'},
            {'name': 'speed', 'min': MIN_SPEED, 'max': MAX_SPEED, 
             'default': DEFAULT_SPEED, 'label': 'Speed'},
            {'name': 'strength', 'min': MIN_STRENGTH, 'max': MAX_STRENGTH, 
             'default': DEFAULT_STRENGTH, 'label': 'Strength'}
        ]
    },
    {
        'name': 'wave',
        'label': 'Wave',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'},
            {'name': 'amplitude', 'min': MIN_AMPLITUDE, 'max': MAX_AMPLITUDE, 
             'default': DEFAULT_AMPLITUDE, 'label': 'Amplitude'},
            {'name': 'frequency', 'min': MIN_FREQUENCY, 'max': MAX_FREQUENCY, 
             'default': DEFAULT_FREQUENCY, 'label': 'Frequency'},
            {'name': 'speed', 'min': MIN_SPEED, 'max': MAX_SPEED, 
             'default': DEFAULT_SPEED, 'label': 'Speed'}
        ]
    },
    {
        'name': 'glitch',
        'label': 'Glitch',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'},
            {'name': 'intensity', 'min': MIN_INTENSITY, 'max': MAX_INTENSITY, 
             'default': DEFAULT_INTENSITY, 'label': 'Intensity'},
            {'name': 'speed', 'min': MIN_SPEED, 'max': MAX_SPEED, 
             'default': DEFAULT_SPEED, 'label': 'Speed'}
        ]
    },
    {
        'name': 'grow',
        'label': 'Grow Masks',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'},
            {'name': 'speed', 'min': MIN_SPEED, 'max': MAX_SPEED, 
             'default': DEFAULT_SPEED, 'label': 'Speed'},
            {'name': 'switch_interval', 'min': 1, 'max': 10, 
             'default': 2, 'label': 'Switch Interval'}
        ]
    },
    {
        'name': 'shrink',
        'label': 'Shrink Masks',
        'params': [
            {'name': 'pixelation', 'min': MIN_PIXELATION, 'max': MAX_PIXELATION, 
             'default': DEFAULT_PIXELATION, 'label': 'Pixelation'},
            {'name': 'speed', 'min': MIN_SPEED, 'max': MAX_SPEED, 
             'default': DEFAULT_SPEED, 'label': 'Speed'},
            {'name': 'strength', 'min': MIN_STRENGTH, 'max': MAX_STRENGTH, 
             'default': DEFAULT_STRENGTH, 'label': 'Strength'}
        ]
    }
]
