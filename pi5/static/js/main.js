// Global variables
let currentEffect = 'none';
let effectParams = {};
let rawCanvas, processedCanvas, rawCtx, processedCtx;
let effects = window.EFFECTS_LIST;
let isFullscreen = false;
let originalDimensions = {
    raw: { width: 640, height: 360 },
    processed: { width: 640, height: 360 }
};

// Canvas setup
function setupCanvas(canvas) {
    canvas.width = 640;
    canvas.height = 360;
    return canvas.getContext('2d');
}

// Stream updates
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
            rawImg.onload = () => {
                rawCtx.imageSmoothingEnabled = false;
                const destWidth = isFullscreen ? rawCanvas.width : originalDimensions.raw.width;
                const destHeight = isFullscreen ? rawCanvas.height : originalDimensions.raw.height;
                rawCtx.drawImage(rawImg, 0, 0, destWidth, destHeight);
            };
            rawImg.src = 'data:image/jpeg;base64,' + data.raw;
            
            const processedImg = new Image();
            processedImg.onload = () => {
                processedCtx.imageSmoothingEnabled = false;
                const destWidth = isFullscreen ? processedCanvas.width : originalDimensions.processed.width;
                const destHeight = isFullscreen ? processedCanvas.height : originalDimensions.processed.height;
                processedCtx.drawImage(processedImg, 0, 0, destWidth, destHeight);
            };
            processedImg.src = 'data:image/jpeg;base64,' + data.processed;
        }
    } catch (error) {
        console.error('Error updating streams:', error);
    }
}

// Effect controls
function setupEffectControls() {
    const buttonContainer = document.getElementById('effect-buttons');
    buttonContainer.innerHTML = '';
    
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
            
            value.textContent = formatParamValue(slider.value, param);
            
            slider.oninput = () => {
                value.textContent = formatParamValue(slider.value, param);
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
    
    document.querySelectorAll('.effect-btn').forEach(btn => {
        const effect = effects.find(e => e.label === btn.textContent);
        btn.classList.toggle('active', effect && effect.name === effectName);
    });
    
    effectParams = {};
    updateParamControls();
    
    fetch(`/effect/${effectName}`, {
        method: 'POST'
    });
}

// Fullscreen functionality
async function toggleFullscreen(streamDiv) {
    const container = streamDiv.closest('.stream');
    const canvas = container.querySelector('canvas');
    const ctx = canvas.getContext('2d');

    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
        try {
            // Store original dimensions
            if (canvas === rawCanvas) {
                originalDimensions.raw = { width: canvas.width, height: canvas.height };
            } else {
                originalDimensions.processed = { width: canvas.width, height: canvas.height };
            }

            // Enter fullscreen
            if (container.requestFullscreen) {
                await container.requestFullscreen();
            } else if (container.webkitRequestFullscreen) {
                await container.webkitRequestFullscreen();
            }

            // Set canvas to screen dimensions
            canvas.width = window.screen.width;
            canvas.height = window.screen.height;

            // Force container to fill screen
            container.style.width = '100vw';
            container.style.height = '100vh';
            container.style.padding = '0';
            container.style.margin = '0';
            container.style.backgroundColor = 'black';

            // Configure canvas for fullscreen
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.maxWidth = 'none';
            canvas.style.maxHeight = 'none';
            canvas.style.margin = '0';
            canvas.style.padding = '0';
            canvas.style.objectFit = 'fill';
            canvas.style.borderRadius = '0';
            canvas.style.border = 'none';

            // Disable smoothing
            ctx.imageSmoothingEnabled = false;
            ctx.webkitImageSmoothingEnabled = false;
            ctx.mozImageSmoothingEnabled = false;
            ctx.msImageSmoothingEnabled = false;

            isFullscreen = true;
            canvas.style.cursor = 'none';

        } catch (err) {
            console.error('Error entering fullscreen:', err);
        }
    } else {
        try {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            }

            // Restore original dimensions
            if (canvas === rawCanvas) {
                canvas.width = originalDimensions.raw.width;
                canvas.height = originalDimensions.raw.height;
            } else {
                canvas.width = originalDimensions.processed.width;
                canvas.height = originalDimensions.processed.height;
            }

            // Reset all styles
            container.style = '';
            canvas.style = '';

            isFullscreen = false;
            canvas.style.cursor = 'pointer';

        } catch (err) {
            console.error('Error exiting fullscreen:', err);
        }
    }
}

// Handle fullscreen change
function handleFullscreenChange() {
    const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
    isFullscreen = !!fullscreenElement;
    
    if (!isFullscreen) {
        const canvases = document.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const container = canvas.closest('.stream');
            
            // Reset dimensions
            if (canvas === rawCanvas) {
                canvas.width = originalDimensions.raw.width;
                canvas.height = originalDimensions.raw.height;
            } else {
                canvas.width = originalDimensions.processed.width;
                canvas.height = originalDimensions.processed.height;
            }

            // Reset styles
            container.style = '';
            canvas.style = '';
            canvas.style.cursor = 'pointer';

            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.webkitImageSmoothingEnabled = false;
            ctx.mozImageSmoothingEnabled = false;
            ctx.msImageSmoothingEnabled = false;
        });
    }
}

// Helper functions
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
            fpsDisplay.className = 'fps-display';
            document.body.appendChild(fpsDisplay);
        }
        fpsDisplay.textContent = `FPS: ${fps.toFixed(1)}`;
        frameCount = 0;
        lastFrameTime = currentTime;
    }
}

function animate() {
    updateFPS();
    requestAnimationFrame(animate);
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize canvases
    rawCanvas = document.getElementById('rawCanvas');
    processedCanvas = document.getElementById('processedCanvas');
    rawCtx = setupCanvas(rawCanvas);
    processedCtx = setupCanvas(processedCanvas);
    
    // Disable smoothing on both contexts
    [rawCtx, processedCtx].forEach(ctx => {
        ctx.imageSmoothingEnabled = false;
        ctx.webkitImageSmoothingEnabled = false;
        ctx.mozImageSmoothingEnabled = false;
        ctx.msImageSmoothingEnabled = false;
    });
    
    // Setup controls
    setupEffectControls();
    
    // Start stream updates
    setInterval(updateStreams, 50);
    
    // Setup 8-bit toggle
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
    
    // Initialize default effect parameters
    const defaultEffect = effects.find(e => e.name === 'none');
    if (defaultEffect && defaultEffect.params) {
        defaultEffect.params.forEach(param => {
            effectParams[param.name] = param.default;
        });
        fetch('/effect_params', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(effectParams)
        });
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        if (key === 'n') setEffect('none');
        else if (key === 'm') setEffect('melt');
        else if (key === 'w') setEffect('wave');
        else if (key === 'g') setEffect('glitch');
        else if (key === '+') setEffect('grow');
        else if (key === '-') setEffect('shrink');
        else if (key === 'Escape' && isFullscreen) {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            }
        }
    });
    
    // Add mouse wheel handler
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

    // Setup fullscreen buttons and listeners
    const canvasContainers = document.querySelectorAll('.canvas-container');
    canvasContainers.forEach(container => {
        const canvas = container.querySelector('canvas');
        const fullscreenBtn = container.querySelector('.fullscreen-button');

        canvas.addEventListener('click', () => {
            toggleFullscreen(container);
        });

        fullscreenBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleFullscreen(container);
        });
    });

    // Add fullscreen change listeners
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    
    // Start animation
    animate();

    // Auto-hide cursor in fullscreen
    let cursorTimeout;
    document.addEventListener('mousemove', () => {
        if (isFullscreen) {
            document.body.style.cursor = 'default';
            clearTimeout(cursorTimeout);
            cursorTimeout = setTimeout(() => {
                if (isFullscreen) {
                    document.body.style.cursor = 'none';
                }
            }, 2000);
        }
    });
});
