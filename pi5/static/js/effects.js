// effects.js
export class EffectsManager {
    constructor() {
        // Get effects list from window or use empty array as fallback
        this.effects = [];
        this.currentEffect = 'none';
        this.effectParams = {};
        this.buttonContainer = null;
        this.paramContainer = null;
        this.callbacks = {
            onEffectChange: null,
            onParamChange: null
        };
    }

    initialize(options = {}) {
        const { onEffectChange, onParamChange, effectsList } = options;
        
        // Set effects list from options or window
        this.effects = effectsList || window.EFFECTS_LIST || [];
        
        this.callbacks.onEffectChange = onEffectChange;
        this.callbacks.onParamChange = onParamChange;
        
        this.buttonContainer = document.getElementById('effect-buttons');
        this.paramContainer = document.getElementById('effect-params');
        
        // Set up initial controls
        this.setupEffectControls();
        this.setupKeyboardShortcuts();
        this.setupMouseWheelHandler();
        
        // Initialize default effect parameters
        const defaultEffect = this.effects.find(e => e.name === 'none');
        if (defaultEffect?.params) {
            defaultEffect.params.forEach(param => {
                this.effectParams[param.name] = param.default;
            });
            this.updateEffectParams();
        }
    }

    setupEffectControls() {
        if (!this.buttonContainer) return;
        
        this.buttonContainer.innerHTML = '';
        
        this.effects.forEach(effect => {
            const button = document.createElement('button');
            button.textContent = effect.label;
            button.className = 'effect-btn';
            if (effect.name === this.currentEffect) {
                button.classList.add('active');
            }
            button.onclick = () => this.setEffect(effect.name);
            this.buttonContainer.appendChild(button);
        });
        
        this.updateParamControls();
    }

    updateParamControls() {
        if (!this.paramContainer) return;
        
        this.paramContainer.innerHTML = '';
        
        const effect = this.effects.find(e => e.name === this.currentEffect);
        if (!effect?.params) return;
        
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
            slider.value = this.effectParams[param.name] || param.default;
            slider.className = 'slider';
            
            value.textContent = this.formatParamValue(slider.value, param);
            
            slider.oninput = () => {
                value.textContent = this.formatParamValue(slider.value, param);
                this.effectParams[param.name] = parseInt(slider.value);
                this.updateEffectParams();
            };
            
            container.appendChild(label);
            container.appendChild(slider);
            container.appendChild(value);
            this.paramContainer.appendChild(container);
        });
    }

    async setEffect(effectName) {
        console.log('Setting effect:', effectName);
        this.currentEffect = effectName;
        
        // Update button states
        document.querySelectorAll('.effect-btn').forEach(btn => {
            const effect = this.effects.find(e => e.label === btn.textContent);
            btn.classList.toggle('active', effect && effect.name === effectName);
        });
        
        // Reset parameters to defaults for the new effect
        const effect = this.effects.find(e => e.name === effectName);
        this.effectParams = {};
        if (effect?.params) {
            effect.params.forEach(param => {
                this.effectParams[param.name] = param.default;
            });
        }
        
        this.updateParamControls();
        
        try {
            // Immediately send both effect change and initial parameters
            await fetch(`/effect/${effectName}`, { method: 'POST' });
            await this.updateEffectParams(); // Send initial parameters
            
            if (this.callbacks.onEffectChange) {
                this.callbacks.onEffectChange(effectName);
            }
        } catch (error) {
            console.error('Error setting effect:', error);
        }
    }

    async updateEffectParams() {
        try {
            await fetch('/effect_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.effectParams)
            });
            
            if (this.callbacks.onParamChange) {
                this.callbacks.onParamChange(this.effectParams);
            }
        } catch (error) {
            console.error('Error updating effect parameters:', error);
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            const key = event.key.toLowerCase();
            const shortcuts = {
                'n': 'none',
                'm': 'melt',
                'w': 'wave',
                'g': 'glitch',
                '+': 'grow',
                '-': 'shrink'
            };
            
            if (shortcuts[key]) {
                this.setEffect(shortcuts[key]);
            }
        });
    }

    setupMouseWheelHandler() {
        document.addEventListener('wheel', (event) => {
            if (event.ctrlKey || event.metaKey) {
                event.preventDefault();
                this.handlePixelationWheel(event);
            }
        }, { passive: false });
    }

    handlePixelationWheel(event) {
        const currentEffect = this.effects.find(e => e.name === this.currentEffect);
        if (currentEffect?.params) {
            const pixelation = this.effectParams['pixelation'] || 6;
            const newValue = event.deltaY > 0 ? 
                Math.min(pixelation + 1, 20) : 
                Math.max(pixelation - 1, 1);
            this.effectParams['pixelation'] = newValue;
            this.updateParamControls();
            this.updateEffectParams();
        }
    }

    formatParamValue(value, param) {
        const formatters = {
            pixelation: value => `${value}px`,
            speed: value => `${value}x`,
            amplitude: value => `${value}px`,
            frequency: value => `${value}hz`,
            strength: value => `${value}x`,
            intensity: value => `${value}x`
        };

        const formatter = formatters[param.name] || (value => value);
        return formatter(value);
    }

    getCurrentEffect() {
        return this.currentEffect;
    }

    getCurrentParams() {
        return { ...this.effectParams };
    }
}

export default EffectsManager;
