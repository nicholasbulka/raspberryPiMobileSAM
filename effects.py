import numpy as np
import cv2
import json
import time
from scipy.ndimage import binary_dilation, binary_erosion
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import random

print("Loading effects module...")

@dataclass
class MaskState:
    mask_index: int
    phase: str  # 'growing' or 'shrinking'
    size: int
    wall_touches: int
    original_mask: np.ndarray

class MaskAnimator:
    def __init__(self):
        self.current_mask_state: Optional[MaskState] = None
        self.masks: List[np.ndarray] = []
        self.mask_metadata: List[Dict[str, Any]] = []
        self.last_switch_time = 0
        self.switch_interval = 2.0  # Time before switching masks

    def update_masks(self, masks: List[np.ndarray], metadata: List[Dict[str, Any]]):
        self.masks = masks
        self.mask_metadata = metadata
        if not self.current_mask_state and masks:
            self.select_new_mask()

    def select_new_mask(self):
        if not self.masks:
            self.current_mask_state = None
            return

        mask_index = random.randint(0, len(self.masks) - 1)
        self.current_mask_state = MaskState(
            mask_index=mask_index,
            phase='growing',
            size=1,
            wall_touches=0,
            original_mask=self.masks[mask_index].copy()
        )
        self.last_switch_time = time.time()

    def check_wall_touches(self, mask: np.ndarray) -> int:
        h, w = mask.shape
        touches = 0
        if np.any(mask[0, :]):  # Top wall
            touches += 1
        if np.any(mask[-1, :]):  # Bottom wall
            touches += 1
        if np.any(mask[:, 0]):  # Left wall
            touches += 1
        if np.any(mask[:, -1]):  # Right wall
            touches += 1
        return touches

    def get_current_state(self) -> Dict[str, Any]:
        if not self.current_mask_state:
            return {"active_mask": None}
        
        mask_meta = self.mask_metadata[self.current_mask_state.mask_index]
        return {
            "active_mask": self.current_mask_state.mask_index,
            "phase": self.current_mask_state.phase,
            "size": self.current_mask_state.size,
            "wall_touches": self.current_mask_state.wall_touches,
            "score": mask_meta.get("score", 0),
            "area": float(np.sum(self.current_mask_state.original_mask)),
            "stability_score": mask_meta.get("stability_score", 0)
        }

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.current_mask_state or not self.masks:
            return frame

        result = frame.copy()
        current_time = time.time()
        
        if current_time - self.last_switch_time > self.switch_interval:
            if self.current_mask_state.phase == 'shrinking' and self.current_mask_state.size <= 1:
                self.select_new_mask()
                return frame

        mask = self.current_mask_state.original_mask.copy()
        
        if self.current_mask_state.phase == 'growing':
            grown_mask = binary_dilation(mask, iterations=self.current_mask_state.size)
            wall_touches = self.check_wall_touches(grown_mask)
            
            if wall_touches >= 3:
                self.current_mask_state.phase = 'shrinking'
            else:
                self.current_mask_state.size += 1
                self.current_mask_state.wall_touches = wall_touches
                result[grown_mask] = np.clip(frame[grown_mask] * 1.2, 0, 255).astype(np.uint8)
                
        else:  # shrinking
            if self.current_mask_state.size > 1:
                self.current_mask_state.size -= 1
                grown_mask = binary_dilation(mask, iterations=self.current_mask_state.size)
                result[grown_mask] = np.clip(frame[grown_mask] * 1.2, 0, 255).astype(np.uint8)

        return result

# Global animator instance
mask_animator = MaskAnimator()

EFFECTS_LIST = [
    {
        'name': 'none',
        'label': 'No Effect',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'}
        ]
    },
    {
        'name': 'melt',
        'label': 'Melt',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 3, 'label': 'Speed'},
            {'name': 'strength', 'min': 1, 'max': 5, 'default': 2, 'label': 'Strength'}
        ]
    },
    {
        'name': 'wave',
        'label': 'Wave',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'},
            {'name': 'amplitude', 'min': 1, 'max': 20, 'default': 10, 'label': 'Amplitude'},
            {'name': 'frequency', 'min': 1, 'max': 10, 'default': 3, 'label': 'Frequency'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'}
        ]
    },
    {
        'name': 'glitch',
        'label': 'Glitch',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'},
            {'name': 'intensity', 'min': 1, 'max': 10, 'default': 5, 'label': 'Intensity'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'}
        ]
    },
    {
        'name': 'grow',
        'label': 'Grow Masks',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'},
            {'name': 'switch_interval', 'min': 1, 'max': 10, 'default': 2, 'label': 'Switch Interval'}
        ]
    },
    {
        'name': 'shrink',
        'label': 'Shrink Masks',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 6, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'},
            {'name': 'strength', 'min': 1, 'max': 5, 'default': 2, 'label': 'Strength'}
        ]
    }
]

def safe_array_access(arr, indices, default_value=0):
    """Safely access array elements with bounds checking"""
    try:
        return arr[indices]
    except IndexError:
        return default_value

def apply_pixelation(frame, pixelation_factor):
    """Apply pixelation effect to frame"""
    try:
        h, w = frame.shape[:2]
        
        # Ensure pixelation factor is reasonable
        pixelation_factor = max(1, min(pixelation_factor, 20))
        
        # Calculate new dimensions
        small_h = max(1, h // pixelation_factor)
        small_w = max(1, w // pixelation_factor)
        
        # Downscale and upscale
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error in pixelation: {e}")
        return frame

def apply_melt(frame, params):
    try:
        pixelation = params.get('pixelation', 6)
        speed = params.get('speed', 3)
        strength = params.get('strength', 2)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = frame.copy()

        offset_time = time.time() * speed
        for x in range(width):
            wave = np.sin(x * 0.1 + offset_time) * strength
            offset = int(abs(wave))
            if offset > 0 and offset < height:
                result[offset:, x] = safe_array_access(result, (slice(None, -offset), x))
                result[:offset, x] = safe_array_access(result, (slice(offset, offset+1), x))

        return result
    except Exception as e:
        print(f"Error in melt effect: {e}")
        return frame

def apply_wave(frame, params):
    try:
        pixelation = params.get('pixelation', 6)
        amplitude = params.get('amplitude', 10)
        frequency = params.get('frequency', 3)
        speed = params.get('speed', 5)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = np.zeros_like(frame)

        offset_time = time.time() * speed
        for y in range(height):
            for x in range(width):
                offset = int(amplitude * np.sin(2 * np.pi * frequency * x / width + offset_time))
                new_y = (y + offset) % height
                result[new_y, x] = frame[y, x]

        return result
    except Exception as e:
        print(f"Error in wave effect: {e}")
        return frame

def apply_glitch(frame, params):
    try:
        pixelation = params.get('pixelation', 6)
        intensity = params.get('intensity', 5)
        speed = params.get('speed', 5)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = frame.copy()

        seed = int(time.time() * speed)
        np.random.seed(seed)

        num_glitches = int(intensity * 5)
        for _ in range(num_glitches):
            h = np.random.randint(5, 40)
            y = np.random.randint(0, height - h)
            offset = np.random.randint(-20, 20)

            if offset != 0:
                if offset > 0:
                    result[y:y+h, offset:] = safe_array_access(result, (slice(y, y+h), slice(None, -offset)))
                else:
                    result[y:y+h, :offset] = safe_array_access(result, (slice(y, y+h), slice(-offset, None)))

            channel = np.random.randint(0, 3)
            result[y:y+h, :, channel] = np.roll(result[y:y+h, :, channel], offset, axis=1)

        return result
    except Exception as e:
        print(f"Error in glitch effect: {e}")
        return frame

def apply_grow_masks(frame, masks, params, metadata=None):
    if masks is None or not masks:
        return frame

    try:
        pixelation = params.get('pixelation', 6)
        switch_interval = params.get('switch_interval', 2)
        
        frame = apply_pixelation(frame, pixelation)
        mask_animator.switch_interval = switch_interval
        
        if metadata is None:
            metadata = [{"score": 1.0} for _ in masks]
        
        mask_animator.update_masks(masks, metadata)
        return mask_animator.process_frame(frame)
        
    except Exception as e:
        print(f"Error in grow_masks effect: {e}")
        return frame

def apply_shrink_masks(frame, masks, params):
    if masks is None:
        return frame

    try:
        pixelation = params.get('pixelation', 6)
        speed = params.get('speed', 5)
        strength = params.get('strength', 2)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = frame.copy()

        time_factor = (np.sin(time.time() * speed) + 1) / 2
        current_strength = int(strength * 2 * time_factor) + 1

        for mask in masks:
            mask_resized = cv2.resize(mask.astype(np.uint8), (width, height)) > 0
            shrunk_mask = binary_erosion(mask_resized, iterations=current_strength)
            result[~shrunk_mask] = np.clip(frame[~shrunk_mask] * 0.8, 0, 255).astype(np.uint8)

        return result
    except Exception as e:
        print(f"Error in shrink_masks effect: {e}")
        return frame
def apply_effect(frame, effect_name, params, masks=None, metadata=None):
    try:
        print(f"Applying effect: {effect_name} with params: {params}")
        
        if effect_name == 'none':
            pixelation = params.get('pixelation', 6)
            return apply_pixelation(frame, pixelation)
        elif effect_name == 'melt':
            return apply_melt(frame, params)
        elif effect_name == 'wave':
            return apply_wave(frame, params)
        elif effect_name == 'glitch':
            return apply_glitch(frame, params)
        elif effect_name == 'grow':
            return apply_grow_masks(frame, masks, params, metadata)
        elif effect_name == 'shrink':
            return apply_shrink_masks(frame, masks, params)
        
        return frame
    except Exception as e:
        print(f"Error applying effect {effect_name}: {e}")
        return frame

def get_javascript_code():
    """Generate JavaScript code for the web interface"""
    return json.dumps(EFFECTS_LIST)

print("Effects module loaded successfully")
