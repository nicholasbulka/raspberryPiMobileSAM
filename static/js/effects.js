export default class EffectsManager {
    constructor(effects) {
        this.effects = effects;
        this.currentEffect = 'none';
        this.effectParams = {};
        this.buttonContainer = null;
        this.paramContainer = null;
        this.debugInfo = {
            maskData: [],
            activeMask: null,
            performanceMetrics: {
                fps: 0,
                totalMasks: 0,
                avgConfidence: 0
            }
        };
        this.lastUpdateTime = performance.now();
        this.frameCount = 0;
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

        // Initialize debug panel
        this.setupDebugPanel();
    }

    setupDebugPanel() {
        // Setup search functionality
        const searchInput = document.getElementById('mask-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterMasks(e.target.value);
            });
        }

        // Setup collapse/expand buttons
        const collapseBtn = document.getElementById('collapse-all');
        const expandBtn = document.getElementById('expand-all');
        
        if (collapseBtn) {
            collapseBtn.addEventListener('click', () => {
                this.toggleAllMasks(false);
            });
        }
        
        if (expandBtn) {
            expandBtn.addEventListener('click', () => {
                this.toggleAllMasks(true);
            });
        }

        // Start debug update loop
        this.startDebugUpdates();
    }

    startDebugUpdates() {
        setInterval(() => {
            this.updateDebugInfo();
        }, 1000); // Update every second
    }

    async updateDebugInfo() {
        try {
            const response = await fetch('/debug_info');
            if (!response.ok) return;
            
            const data = await response.json();
            this.debugInfo = data;
            
            this.updateDebugDisplay();
            this.updatePerformanceMetrics();
            
        } catch (error) {
            console.error('Error updating debug info:', error);
        }
    }

    updateDebugDisplay() {
        // Update active mask information
        const activeMaskInfo = document.getElementById('active-mask-info');
        if (activeMaskInfo && this.debugInfo.activeMask) {
            activeMaskInfo.innerHTML = this.formatJSON(this.debugInfo.activeMask);
        }

        // Update all masks information
        const allMasksInfo = document.getElementById('all-masks-info');
        if (allMasksInfo && this.debugInfo.maskData) {
            allMasksInfo.innerHTML = this.formatJSON(this.debugInfo.maskData);
        }
    }

    updatePerformanceMetrics() {
        const currentTime = performance.now();
        const elapsed = currentTime - this.lastUpdateTime;
        
        if (elapsed >= 1000) {
            const fps = (this.frameCount / elapsed) * 1000;
            
            document.getElementById('processing-fps').textContent = fps.toFixed(1);
            document.getElementById('total-masks').textContent = this.debugInfo.maskData.length;
            
            const avgConfidence = this.debugInfo.maskData.reduce((acc, mask) => 
                acc + (mask.score || 0), 0) / this.debugInfo.maskData.length;
            document.getElementById('avg-confidence').textContent = 
                (avgConfidence * 100).toFixed(1) + '%';

            this.frameCount = 0;
            this.lastUpdateTime = currentTime;
        }
        
        this.frameCount++;
    }

    formatJSON(data, level = 0) {
        const indent = '  '.repeat(level);
        
        if (Array.isArray(data)) {
            if (data.length === 0) return '[]';
            
            return `[\n${data.map(item => 
                `${indent}  ${this.formatJSON(item, level + 1)}`
            ).join(',\n')}\n${indent}]`;
        }
        
        if (data === null) return 'null';
        if (typeof data !== 'object') return JSON.stringify(data);
        
        const entries = Object.entries(data).map(([key, value]) => {
            if (typeof value === 'number') {
                value = Number(value.toFixed(4));
            }
            return `${indent}  "${key}": ${this.formatJSON(value, level + 1)}`;
        });
        
        if (entries.length === 0) return '{}';
        return `{\n${entries.join(',\n')}\n${indent}}`;
    }

    filterMasks(searchTerm) {
        const allMasks = document.querySelectorAll('.mask-entry');
        searchTerm = searchTerm.toLowerCase();
        
        allMasks.forEach(mask => {
            const text = mask.textContent.toLowerCase();
            mask.style.display = text.includes(searchTerm) ? '' : 'none';
        });
    }

    toggleAllMasks(expand) {
        const maskEntries = document.querySelectorAll('.mask-entry');
        maskEntries.forEach(entry => {
            const content = entry.querySelector('.mask-content');
            if (content) {
                content.style.display = expand ? 'block' : 'none';
            }
        });
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
	async fetchMaskInfo() {
        try {
            const response = await fetch('/mask_info');
            if (!response.ok) return;

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching mask info:', error);
            return null;
        }
    }

    createMaskEntry(mask, index) {
        const entry = document.createElement('div');
        entry.className = 'mask-entry';

        const header = document.createElement('div');
        header.className = 'mask-header';
        header.innerHTML = `
            <span class="mask-index">Mask ${index + 1}</span>
            <span class="mask-score">Score: ${(mask.score * 100).toFixed(1)}%</span>
            <button class="mask-toggle">▼</button>
        `;

        const content = document.createElement('div');
        content.className = 'mask-content';
        content.innerHTML = this.formatJSON(mask);

        header.querySelector('.mask-toggle').onclick = () => {
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
            header.querySelector('.mask-toggle').textContent =
                content.style.display === 'none' ? '▶' : '▼';
        };

        entry.appendChild(header);
        entry.appendChild(content);
        return entry;
    }

    updateActiveMaskVisualization() {
        if (!this.debugInfo.activeMask) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const mask = this.debugInfo.activeMask;

        // Set canvas size to match mask dimensions
        canvas.width = mask.width;
        canvas.height = mask.height;

        // Draw mask outline
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.beginPath();

        // Draw mask data
        const imageData = ctx.createImageData(mask.width, mask.height);
        for (let i = 0; i < mask.data.length; i++) {
            const idx = i * 4;
            if (mask.data[i]) {
                imageData.data[idx] = 0;     // R
                imageData.data[idx + 1] = 255; // G
                imageData.data[idx + 2] = 0;   // B
                imageData.data[idx + 3] = 128; // A
            }
        }
        ctx.putImageData(imageData, 0, 0);

        // Update visualization in debug panel
        const visualizationContainer = document.getElementById('mask-visualization');
        if (visualizationContainer) {
            visualizationContainer.innerHTML = '';
            visualizationContainer.appendChild(canvas);
        }
    }

    // Utility function to create collapsible JSON view
    createCollapsibleJSON(data, level = 0) {
        const container = document.createElement('div');
        container.className = 'json-container';

        const toggle = document.createElement('span');
        toggle.className = 'json-toggle';
        toggle.textContent = '▼';

        const content = document.createElement('div');
        content.className = 'json-content';
        content.innerHTML = this.formatJSON(data);

        toggle.onclick = () => {
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
            toggle.textContent = content.style.display === 'none' ? '▶' : '▼';
        };

        container.appendChild(toggle);
        container.appendChild(content);
        return container;
    }

    // Helper function to format numbers nicely
    formatNumber(num) {
        if (typeof num !== 'number') return num;
        if (Number.isInteger(num)) return num.toString();
        return num.toFixed(4);
    }

    // Helper function to check if a string contains a search term
    containsSearchTerm(str, searchTerm) {
        return str.toLowerCase().includes(searchTerm.toLowerCase());
    }

    // Update debug panel layout
    updateDebugLayout() {
        const debugPanel = document.getElementById('debug-panel');
        if (!debugPanel) return;

        const viewportHeight = window.innerHeight;
        const headerHeight = document.querySelector('.debug-header')?.offsetHeight || 0;
        const padding = 40; // Account for padding and margins

        debugPanel.style.maxHeight = `${viewportHeight - headerHeight - padding}px`;
    }

    // Initialize debug panel resize handling
    initializeDebugResize() {
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        const debugPanel = document.getElementById('debug-panel');
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'debug-resize-handle';
        debugPanel.appendChild(resizeHandle);

        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.pageX;
            startWidth = debugPanel.offsetWidth;
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const width = startWidth - (e.pageX - startX);
            debugPanel.style.width = `${Math.max(300, Math.min(800, width))}px`;
        });

        document.addEventListener('mouseup', () => {
            isResizing = false;
        });
    }
}
