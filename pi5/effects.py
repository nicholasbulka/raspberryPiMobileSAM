import numpy as np
import cv2
import json
import time
from scipy.ndimage import binary_dilation, binary_erosion

print("Loading effects module...")

EFFECTS_LIST = [
    {
        'name': 'none',
        'label': 'No Effect',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'}
        ]
    },
    {
        'name': 'melt',
        'label': 'Melt',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 3, 'label': 'Speed'},
            {'name': 'strength', 'min': 1, 'max': 5, 'default': 2, 'label': 'Strength'}
        ]
    },
    {
        'name': 'wave',
        'label': 'Wave',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'},
            {'name': 'amplitude', 'min': 1, 'max': 20, 'default': 10, 'label': 'Amplitude'},
            {'name': 'frequency', 'min': 1, 'max': 10, 'default': 3, 'label': 'Frequency'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'}
        ]
    },
    {
        'name': 'glitch',
        'label': 'Glitch',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'},
            {'name': 'intensity', 'min': 1, 'max': 10, 'default': 5, 'label': 'Intensity'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'}
        ]
    },
    {
        'name': 'grow',
        'label': 'Grow Masks',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'},
            {'name': 'strength', 'min': 1, 'max': 5, 'default': 2, 'label': 'Strength'}
        ]
    },
    {
        'name': 'shrink',
        'label': 'Shrink Masks',
        'params': [
            {'name': 'pixelation', 'min': 1, 'max': 20, 'default': 1, 'label': 'Pixelation'},
            {'name': 'speed', 'min': 1, 'max': 10, 'default': 5, 'label': 'Speed'},
            {'name': 'strength', 'min': 1, 'max': 5, 'default': 2, 'label': 'Strength'}
        ]
    }
]


def safe_array_access(arr, indices, default_value=None):
    """Safely access array elements with bounds checking and error handling.
    
    Args:
        arr: numpy array to access
        indices: tuple of slice objects or indices
        default_value: value to return if access fails (default None)
    
    Returns:
        Array slice if successful, default_value if access would be invalid
    """
    try:
        # Validate array
        if arr is None or not isinstance(arr, np.ndarray):
            return default_value
            
        # Convert single index to tuple
        if not isinstance(indices, tuple):
            indices = (indices,)
            
        # Validate each index/slice
        for idx, dim in zip(indices, arr.shape):
            if isinstance(idx, slice):
                # For slices, check start and stop if they're not None
                if idx.start is not None and (idx.start < -dim or idx.start > dim):
                    return default_value
                if idx.stop is not None and (idx.stop < -dim or idx.stop > dim):
                    return default_value
            else:
                # For direct indices, check bounds
                if idx < -dim or idx >= dim:
                    return default_value
        
        # If all checks pass, return the slice
        return arr[indices]
        
    except Exception as e:
        print(f"Error in safe_array_access: {e}")
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

def apply_grow_masks(frame, masks, params):
    if masks is None or not isinstance(masks, (list, np.ndarray)) or len(masks) == 0:
        return frame

    try:
        pixelation = params.get('pixelation', 6)
        speed = params.get('speed', 5)
        strength = params.get('strength', 2)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = frame.copy()

        # Improved time-based variation with smoother transitions
        time_factor = (np.sin(time.time() * max(0.5, speed/2)) + 1) / 2  # 0 to 1, slower variation
        # Scale strength based on image size
        max_strength = min(20, int(min(height, width) * 0.1))  # Max 10% of smallest dimension
        current_strength = int(max(1, min(max_strength, strength * 3 * time_factor)))

        intensity = 1.0 + (strength * 0.2)  # Dynamic intensity based on strength

        for mask in masks:
            if mask.size == 0:
                continue
                
            # Ensure mask is proper size and binary
            mask_resized = cv2.resize(mask.astype(np.uint8), (width, height))
            mask_binary = mask_resized > 0
            
            # Apply dilation with error checking
            try:
                grown_mask = binary_dilation(mask_binary, iterations=current_strength)
                
                # Apply effect with bounds checking
                mask_indices = np.where(grown_mask)
                if len(mask_indices[0]) > 0:  # Check if mask has any True values
                    result[mask_indices] = np.clip(
                        frame[mask_indices] * intensity, 
                        0, 
                        255
                    ).astype(np.uint8)
            except Exception as e:
                print(f"Error in mask processing: {e}")
                continue

        return result
    except Exception as e:
        print(f"Error in grow_masks effect: {e}")
        return frame

def apply_glitch(frame, params):
    try:
        pixelation = params.get('pixelation', 6)
        intensity = params.get('intensity', 5)
        speed = params.get('speed', 5)

        frame = apply_pixelation(frame, pixelation)
        height, width = frame.shape[:2]
        result = frame.copy()

        # Scale glitch parameters based on image dimensions
        max_height = int(height * 0.2)  # Max 20% of image height
        min_height = max(3, int(height * 0.01))  # Min 1% of image height
        max_offset = int(width * 0.1)  # Max 10% of image width

        # Improved time-based seed for more varied patterns
        base_seed = int(time.time() * speed)
        fine_seed = int((time.time() % 1) * 1000)  # Sub-second variation
        np.random.seed(base_seed + fine_seed)

        # Scale number of glitches with intensity and frame size
        base_glitches = max(1, int(intensity * 2))  # Minimum 1 glitch
        size_factor = (width * height) / (640 * 360)  # Reference size
        num_glitches = int(base_glitches * np.sqrt(size_factor))

        # Create glitches
        for _ in range(num_glitches):
            # Randomly vary glitch parameters
            h = np.random.randint(min_height, max_height)
            y = np.random.randint(0, max(1, height - h))
            offset = np.random.randint(-max_offset, max_offset)

            if offset != 0:
                # Apply horizontal offset with safe bounds
                if offset > 0:
                    source_slice = safe_array_access(result, (slice(y, y+h), slice(None, -offset)))
                    if source_slice is not None:
                        result[y:y+h, offset:] = source_slice
                else:
                    source_slice = safe_array_access(result, (slice(y, y+h), slice(-offset, None)))
                    if source_slice is not None:
                        result[y:y+h, :offset] = source_slice

                # Apply color channel shift with varying intensity
                channel = np.random.randint(0, 3)
                shift_intensity = np.random.uniform(0.5, 1.5)
                channel_data = result[y:y+h, :, channel].copy()
                result[y:y+h, :, channel] = np.clip(
                    np.roll(channel_data * shift_intensity, offset, axis=1),
                    0,
                    255
                ).astype(np.uint8)

                # Random chance for additional effects
                if np.random.random() < 0.3:  # 30% chance
                    # Add noise to glitched region
                    noise = np.random.normal(0, 10, result[y:y+h, :, channel].shape)
                    result[y:y+h, :, channel] = np.clip(
                        result[y:y+h, :, channel] + noise,
                        0,
                        255
                    ).astype(np.uint8)

        return result
    except Exception as e:
        print(f"Error in glitch effect: {e}")
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

        # Time-based variation
        time_factor = (np.sin(time.time() * speed) + 1) / 2  # 0 to 1
        current_strength = int(strength * 2 * time_factor) + 1

        for mask in masks:
            mask_resized = cv2.resize(mask.astype(np.uint8), (width, height)) > 0
            shrunk_mask = binary_erosion(mask_resized, iterations=current_strength)
            result[~shrunk_mask] = np.clip(frame[~shrunk_mask] * 0.8, 0, 255).astype(np.uint8)

        return result
    except Exception as e:
        print(f"Error in shrink_masks effect: {e}")
        return frame

def apply_effect(frame, effect_name, params, masks=None):
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
            return apply_grow_masks(frame, masks, params)
        elif effect_name == 'shrink':
            return apply_shrink_masks(frame, masks, params)
        
        return frame
    except Exception as e:
        print(f"Error applying effect {effect_name}: {e}")
        return frame

def get_javascript_code():
    """Generate JavaScript code for the web interface"""
    js_code = '''
let currentEffect = 'none';
let effectParams = {};
let rawCanvas, processedCanvas, rawCtx, processedCtx;
let effects = ''' + json.dumps(EFFECTS_LIST) + ''';

function setupCanvas(canvas) {
    canvas.width = 640;
    canvas.height = 360;
    return canvas.getContext('2d');
}

async function updateStreams() {
    try {
        const response = await fetch('/stream');
        if (!response.ok) {
            console.error('Stream response not ok:', response.status);
            return;
        }
        
        const data = await response.json();
        
        if (data.raw && data.processed) {
            const rawImg = new Image();
            rawImg.onload = () => rawCtx.drawImage(rawImg, 0, 0);
            rawImg.src = 'data:image/jpeg;base64,' + data.raw;
            
            const processedImg = new Image();
            processedImg.onload = () => processedCtx.drawImage(processedImg, 0, 0);
            processedImg.src = 'data:image/jpeg;base64,' + data.processed;
        }
    } catch (error) {
        console.error('Error updating streams:', error);
    }
}

function setupEffectControls() {
    const buttonContainer = document.getElementById('effect-buttons');
    buttonContainer.innerHTML = ''; // Clear existing buttons
    
    effects.forEach(effect => {
        const button = document.createElement('button');
        button.textContent = effect.label;
        button.className = 'effect-btn';
        if (effect.name === currentEffect) {
            button.classList.add('active');
        }
        button.onclick = () => setEffect(effect.name);
        buttonContainer.appendChild(button);
    });
    
    updateParamControls();
}

function updateParamControls() {
    const paramContainer = document.getElementById('effect-params');
    paramContainer.innerHTML = '';
    
    const effect = effects.find(e => e.name === currentEffect);
    if (!effect) return;
    
    if (effect.params) {
        effect.params.forEach(param => {
            const container = document.createElement('div');
            container.className = 'slider-container';
            
            const label = document.createElement('label');
            label.textContent = param.label;
            
            const value = document.createElement('span');
            value.style.marginLeft = '10px';
            value.style.minWidth = '30px';
            value.style.display = 'inline-block';
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = param.min;
            slider.max = param.max;
            slider.value = effectParams[param.name] || param.default;
            slider.className = 'slider';
            
            value.textContent = slider.value;
            
            slider.oninput = () => {
                value.textContent = slider.value;
                effectParams[param.name] = parseInt(slider.value);
                fetch('/effect_params', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(effectParams)
                });
            };
            
            container.appendChild(label);
            container.appendChild(slider);
            container.appendChild(value);
            paramContainer.appendChild(container);
        });
    }
}

function setEffect(effectName) {
    console.log('Setting effect:', effectName);
    currentEffect = effectName;
    
    // Update button states
    document.querySelectorAll('.effect-btn').forEach(btn => {
        const effect = effects.find(e => e.label === btn.textContent);
        btn.classList.toggle('active', effect && effect.name === effectName);
    });
    
    // Reset parameters
    effectParams = {};
    updateParamControls();
    
    // Notify server
    fetch(`/effect/${effectName}`, {
        method: 'POST'
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize canvases
    rawCanvas = document.getElementById('rawCanvas');
    processedCanvas = document.getElementById('processedCanvas');
    rawCtx = setupCanvas(rawCanvas);
processedCtx = setupCanvas(processedCanvas);
    
    // Setup controls
    setupEffectControls();
    
    // Start stream updates
    setInterval(updateStreams, 50);  // Increased update rate for smoother display
    
    // Add event listener for 8-bit toggle button
    const toggle8bitBtn = document.getElementById('toggle-8bit');
    if (toggle8bitBtn) {
        toggle8bitBtn.addEventListener('click', () => {
            effectParams['8bit'] = !effectParams['8bit'];
            toggle8bitBtn.classList.toggle('active', effectParams['8bit']);
            fetch('/effect_params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(effectParams)
            });
        });
    }
    
    // Initialize parameters for the default effect
    const defaultEffect = effects.find(e => e.name === 'none');
    if (defaultEffect && defaultEffect.params) {
        defaultEffect.params.forEach(param => {
            effectParams[param.name] = param.default;
        });
        // Send initial parameters to server
        fetch('/effect_params', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(effectParams)
        });
    }
    
    // Add keyboard shortcuts for effects
    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        if (key === 'n') setEffect('none');
        else if (key === 'm') setEffect('melt');
        else if (key === 'w') setEffect('wave');
        else if (key === 'g') setEffect('glitch');
        else if (key === '+') setEffect('grow');
        else if (key === '-') setEffect('shrink');
    });
    
    // Add mouse wheel handler for adjusting pixelation
    document.addEventListener('wheel', (event) => {
        if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            const currentEffect = effects.find(e => e.name === currentEffect);
            if (currentEffect && currentEffect.params) {
                const pixelation = effectParams['pixelation'] || 6;
                const newValue = event.deltaY > 0 ? 
                    Math.min(pixelation + 1, 20) : 
                    Math.max(pixelation - 1, 1);
                effectParams['pixelation'] = newValue;
                updateParamControls();
                fetch('/effect_params', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(effectParams)
                });
            }
        }
    }, { passive: false });
});

// Helper function to format parameter values
function formatParamValue(value, param) {
    if (param.name === 'pixelation') return value + 'px';
    if (param.name.includes('speed')) return value + 'x';
    if (param.name === 'amplitude') return value + 'px';
    if (param.name === 'frequency') return value + 'hz';
    return value;
}

// Performance monitoring
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsDisplay = null;

function updateFPS() {
    frameCount++;
    const currentTime = performance.now();
    const elapsed = currentTime - lastFrameTime;
    
    if (elapsed >= 1000) {
        const fps = (frameCount / elapsed) * 1000;
        if (!fpsDisplay) {
            fpsDisplay = document.createElement('div');
            fpsDisplay.style.position = 'fixed';
            fpsDisplay.style.top = '10px';
            fpsDisplay.style.right = '10px';
            fpsDisplay.style.background = 'rgba(0,0,0,0.5)';
            fpsDisplay.style.color = 'white';
            fpsDisplay.style.padding = '5px';
            fpsDisplay.style.borderRadius = '3px';
            document.body.appendChild(fpsDisplay);
        }
        fpsDisplay.textContent = `FPS: ${fps.toFixed(1)}`;
        frameCount = 0;
        lastFrameTime = currentTime;
    }
}

// Add updateFPS to the animation loop
function animate() {
    updateFPS();
    requestAnimationFrame(animate);
}
animate();
''';
    return js_code

print("Effects module loaded successfully")
