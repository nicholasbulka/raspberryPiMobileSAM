class App {
    constructor(effects) {
        this.effects = effects;
        this.streamManager = null;
        this.effectsManager = null;
        this.fullscreenManager = null;
        this.debugRefreshRate = 5000;
        this.isDebugEnabled = false;
        this.debugRefreshTimer = null;
        this.isRawFeedVisible = true;
        this.is8BitMode = false;
        this.initialized = false;
    }
 
    async initialize() {
        try {
            // Import managers
            const StreamManager = (await import('./stream.js')).default;
            const EffectsManager = (await import('./effects.js')).default;
            const FullscreenManager = (await import('./fullscreen.js')).default;
     
            // Create instances
            this.streamManager = new StreamManager();
            this.effectsManager = new EffectsManager(this.effects);
            this.fullscreenManager = new FullscreenManager();
     
            // Initialize stream manager
            const streamInitialized = this.streamManager.initialize();
            if (!streamInitialized) {
                console.error('Failed to initialize stream manager');
                return null;
            }

            // Initialize effects manager
            this.effectsManager.initialize();
     
            // Set up handlers
            this.setupKeyboardShortcuts();
            this.setupMouseWheelHandler();
            this.initializeDebug();
            this.initializeGlobalControls();

            this.initialized = true;
            return this;
        } catch (error) {
            console.error('Error during initialization:', error);
            return null;
        }
    }

    initializeGlobalControls() {
        // 8-bit mode toggle
        const toggle8BitBtn = document.getElementById('toggle-8bit');
        if (toggle8BitBtn) {
            toggle8BitBtn.addEventListener('click', () => {
                this.is8BitMode = !this.is8BitMode;
                toggle8BitBtn.classList.toggle('active', this.is8BitMode);
                if (this.effectsManager) {
                    this.effectsManager.setEffectParam('8bit', this.is8BitMode);
                }
            });
        }

        // Raw feed toggle
        const toggleRawBtn = document.getElementById('toggle-raw');
        if (toggleRawBtn) {
            toggleRawBtn.addEventListener('click', () => {
                this.isRawFeedVisible = !this.isRawFeedVisible;
                toggleRawBtn.classList.toggle('active', this.isRawFeedVisible);
                const rawFeed = document.querySelector('.stream.raw');
                if (rawFeed) {
                    rawFeed.style.display = this.isRawFeedVisible ? 'block' : 'none';
                }
            });
        }

        // Debug panel toggle
        const debugToggle = document.getElementById('toggle-debug');
        if (debugToggle) {
            debugToggle.addEventListener('click', () => this.toggleDebugPanel());
        }
    }
 
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            if (!this.effectsManager) return;

            const key = event.key.toLowerCase();
            if (key === 'n') this.effectsManager.setEffect('none');
            else if (key === 'b') this.effectsManager.setEffect('blur');
            else if (key === 'p') this.effectsManager.setEffect('pixelate');
            else if (key === 'o') this.effectsManager.setEffect('outline');
            else if (key === 'd') this.toggleDebugPanel();
            else if (key === 'escape') this.handleEscapeKey();
            else if (key === '8') document.getElementById('toggle-8bit')?.click();
            else if (key === 'r') document.getElementById('toggle-raw')?.click();
        });
    }
 
    setupMouseWheelHandler() {
        document.addEventListener('wheel', (event) => {
            if (!this.effectsManager) return;

            if (event.ctrlKey || event.metaKey) {
                event.preventDefault();
                this.effectsManager.handlePixelationWheel(event);
            }
        }, { passive: false });
    }
 
    initializeDebug() {
        const closeDebug = document.getElementById('close-debug');
        if (closeDebug) {
            closeDebug.addEventListener('click', () => this.toggleDebugPanel());
        }
 
        // Initialize debug search functionality
        const searchInput = document.getElementById('mask-search');
        if (searchInput && this.effectsManager) {
            searchInput.addEventListener('input', (e) => {
                this.effectsManager.filterMasks(e.target.value);
            });
        }
 
        // Initialize collapse/expand buttons
        const collapseBtn = document.getElementById('collapse-all');
        const expandBtn = document.getElementById('expand-all');
        
        if (collapseBtn && this.effectsManager) {
            collapseBtn.addEventListener('click', () => {
                this.effectsManager.toggleAllMasks(false);
            });
        }
        
        if (expandBtn && this.effectsManager) {
            expandBtn.addEventListener('click', () => {
                this.effectsManager.toggleAllMasks(true);
            });
        }
    }
 
    toggleDebugPanel() {
        const debugPanel = document.getElementById('debug-panel');
        const debugToggle = document.getElementById('toggle-debug');
        
        if (debugPanel) {
            this.isDebugEnabled = !debugPanel.classList.contains('active');
            debugPanel.classList.toggle('active');
            debugToggle?.classList.toggle('active');
            debugPanel.classList.toggle('hidden', !this.isDebugEnabled);
            
            if (this.isDebugEnabled) {
                this.refreshDebugInfo();
                this.startDebugRefresh();
            } else {
                this.stopDebugRefresh();
            }
        }
    }
 
    startDebugRefresh() {
        this.stopDebugRefresh();  // Clear any existing timer
        
        if (this.isDebugEnabled) {
            this.debugRefreshTimer = setInterval(() => {
                this.refreshDebugInfo();
            }, this.debugRefreshRate);
        }
    }
 
    stopDebugRefresh() {
        if (this.debugRefreshTimer) {
            clearInterval(this.debugRefreshTimer);
            this.debugRefreshTimer = null;
        }
    }
 
    async refreshDebugInfo() {
        if (!this.isDebugEnabled) return;

        try {
            const response = await fetch('/debug_info');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const debugInfo = await response.json();
            this.updateDebugDisplay(debugInfo);
        } catch (error) {
            console.error('Error refreshing debug info:', error);
        }
    }
 
    updateDebugDisplay(debugInfo) {
        if (!debugInfo) return;

        // Helper function for safely setting metric values
        const setMetricValue = (id, value, format = null) => {
            const element = document.getElementById(id);
            if (element && value !== undefined && value !== null) {
                element.textContent = format ? format(value) : value;
            }
        };
 
        // Update performance metrics
        if (debugInfo.processor) {
            setMetricValue('processing-fps', debugInfo.processor.fps, 
                v => typeof v === 'number' ? v.toFixed(1) : '-');
        }
 
        if (debugInfo.model) {
            setMetricValue('total-masks', debugInfo.model.total_masks);
            setMetricValue('avg-confidence', debugInfo.model.avg_confidence, 
                v => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '-');
        }
 
        if (debugInfo.system) {
            setMetricValue('cpu-temp', debugInfo.system.cpu?.temperature, 
                v => typeof v === 'number' ? v.toFixed(1) + '°C' : '-');
            setMetricValue('memory-usage', debugInfo.system.memory?.percent, 
                v => typeof v === 'number' ? v.toFixed(1) + '%' : '-');
        }
 
        // Update active mask information
        const activeMaskInfo = document.getElementById('active-mask-info');
        if (activeMaskInfo && debugInfo.effects?.debug?.active_mask && this.effectsManager) {
            activeMaskInfo.innerHTML = this.effectsManager.formatJSON(
                debugInfo.effects.debug.active_mask
            );
        }
 
        // Update all masks info
        const allMasksInfo = document.getElementById('all-masks-info');
        if (allMasksInfo && debugInfo.model?.masks && this.effectsManager) {
            allMasksInfo.innerHTML = this.effectsManager.formatJSON(debugInfo.model.masks);
        }

        // Update effect parameters info
        const effectDebugInfo = document.getElementById('effect-debug-info');
        if (effectDebugInfo && debugInfo.effects && this.effectsManager) {
            effectDebugInfo.innerHTML = this.effectsManager.formatJSON({
                current_effect: debugInfo.effects.current_effect,
                parameters: debugInfo.effects.params
            });
        }
    }
 
    handleEscapeKey() {
        if (this.isDebugEnabled) {
            this.toggleDebugPanel();
        }
        if (this.fullscreenManager) {
            this.fullscreenManager.handleEscape();
        }
    }
 
    start() {
        if (!this.initialized || !this.streamManager) {
            console.error('Cannot start: App not properly initialized');
            return;
        }
        this.streamManager.startStreaming();
    }
}
 
// Export initialization function
export async function initializeApp(effects) {
    try {
        const app = new App(effects);
        const initialized = await app.initialize();
        if (!initialized) {
            throw new Error('Failed to initialize app');
        }
        return initialized;
    } catch (error) {
        console.error('Error initializing app:', error);
        return null;
    }
}