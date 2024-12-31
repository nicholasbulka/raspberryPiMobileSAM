export default class EffectsManager {
    constructor(effects) {
        this.effects = effects;
        this.currentEffect = 'none';
        this.effectParams = {};
        this.buttonContainer = null;
        this.paramContainer = null;
    }

    initialize() {
        this.buttonContainer = document.getElementById('effect-buttons');
        this.paramContainer = document.getElementById('effect-params');
        
        // Set up initial controls
        this.setupEffectControls();
        
        // Initialize default effect parameters
        const defaultEffect = this.effects.find(e => e.name === 'none');
        if (defaultEffect && defaultEffect.params) {
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
    }

    async setEffect(effectName) {
        console.log('Setting effect:', effectName);
        this.currentEffect = effectName;
        
        // Update button states
        document.querySelectorAll('.effect-btn').forEach(btn => {
            const effect = this.effects.find(e => e.label === btn.textContent);
            btn.classList.toggle('active', effect && effect.name === effectName);
        });
        
        // Reset parameters
        this.effectParams = {};
        this.updateParamControls();
        
        try {
            const response = await fetch(`/effect/${effectName}`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to set effect');
        } catch (error) {
            console.error('Error setting effect:', error);
        }
    }

    setEffectParam(name, value) {
        this.effectParams[name] = value;
        this.updateEffectParams();
    }

    async updateEffectParams() {
        try {
            const response = await fetch('/effect_params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(this.effectParams)
            });
            if (!response.ok) throw new Error('Failed to update effect parameters');
        } catch (error) {
            console.error('Error updating effect parameters:', error);
        }
    }

    handlePixelationWheel(event) {
        const currentEffect = this.effects.find(e => e.name === this.currentEffect);
        if (currentEffect && currentEffect.params) {
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
        if (param.name === 'pixelation') return value + 'px';
        if (param.name.includes('speed')) return value + 'x';
        if (param.name === 'amplitude') return value + 'px';
        if (param.name === 'frequency') return value + 'hz';
        return value;
    }
}
