from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
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
    """Tracks state of animated masks"""
    mask_index: int
    phase: str
    size: int
    wall_touches: int
    original_mask: np.ndarray
    start_time: float
    transition_duration: float

class EffectParameters:
    """Handles effect parameters with validation and type conversion"""

    # Parameter bounds and defaults
    PARAMETER_SPECS = {
        'pixelation': {
            'min': MIN_PIXELATION,
            'max': MAX_PIXELATION,
            'default': DEFAULT_PIXELATION,
            'type': int
        },
        'speed': {
            'min': MIN_SPEED,
            'max': MAX_SPEED,
            'default': DEFAULT_SPEED,
            'type': int
        },
        'strength': {
            'min': MIN_STRENGTH,
            'max': MAX_STRENGTH,
            'default': DEFAULT_STRENGTH,
            'type': int
        },
        'intensity': {
            'min': MIN_INTENSITY,
            'max': MAX_INTENSITY,
            'default': DEFAULT_INTENSITY,
            'type': int
        },
        'amplitude': {
            'min': MIN_AMPLITUDE,
            'max': MAX_AMPLITUDE,
            'default': DEFAULT_AMPLITUDE,
            'type': int
        },
        'frequency': {
            'min': MIN_FREQUENCY,
            'max': MAX_FREQUENCY,
            'default': DEFAULT_FREQUENCY,
            'type': int
        },
        'switch_interval': {
            'min': 1,
            'max': 10,
            'default': 2,
            'type': float
        }
    }

    def __init__(self, params_dict: Dict[str, Any]):
        """Initialize parameters with validation"""
        self.pixelation = max(MIN_PIXELATION, min(params_dict.get('pixelation', DEFAULT_PIXELATION), MAX_PIXELATION))
        self.speed = max(MIN_SPEED, min(params_dict.get('speed', DEFAULT_SPEED), MAX_SPEED))
        self.strength = max(MIN_STRENGTH, min(params_dict.get('strength', DEFAULT_STRENGTH), MAX_STRENGTH))
        self.intensity = max(MIN_INTENSITY, min(params_dict.get('intensity', DEFAULT_INTENSITY), MAX_INTENSITY))
        self.amplitude = max(MIN_AMPLITUDE, min(params_dict.get('amplitude', DEFAULT_AMPLITUDE), MAX_AMPLITUDE))
        self.frequency = max(MIN_FREQUENCY, min(params_dict.get('frequency', DEFAULT_FREQUENCY), MAX_FREQUENCY))
        self.switch_interval = max(1, min(params_dict.get('switch_interval', 2), 10))
        self._original_params = params_dict

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get parameter value with fallback default"""
        return self._original_params.get(name, default)

    def get_all_params(self) -> Dict[str, Any]:
        """Get all parameters as dictionary"""
        return {
            'pixelation': self.pixelation,
            'speed': self.speed,
            'strength': self.strength,
            'intensity': self.intensity,
            'amplitude': self.amplitude,
            'frequency': self.frequency,
            'switch_interval': self.switch_interval,
            **self._original_params
        }

class MaskAnimator:
    """Manages mask animations and transitions"""
    
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
        """Update list of available masks and metadata"""
        self.masks = masks
        self.mask_metadata = metadata
        self.performance_metrics['num_masks'] = len(masks)
        
        if not self.current_mask_state and masks:
            self.select_new_mask()

    def select_new_mask(self) -> None:
        """Select and initialize a new mask for animation"""
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
        """Count how many edges the mask touches"""
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
        """Get current mask and performance information"""
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

# Constants for effects list
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