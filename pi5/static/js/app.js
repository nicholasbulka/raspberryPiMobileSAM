class App {
    constructor(effects) {
        this.effects = effects;
        this.streamManager = null;
        this.effectsManager = null;
        this.fullscreenManager = null;
    }

    initialize() {
        // Initialize managers
        this.streamManager = new StreamManager();
        this.effectsManager = new EffectsManager(this.effects);
        this.fullscreenManager = new FullscreenManager();

        // Set up keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Set up mouse wheel handler
        this.setupMouseWheelHandler();

        return this;
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            const key = event.key.toLowerCase();
            if (key === 'n') this.effectsManager.setEffect('none');
            else if (key === 'm') this.effectsManager.setEffect('melt');
            else if (key === 'w') this.effectsManager.setEffect('wave');
            else if (key === 'g') this.effectsManager.setEffect('glitch');
            else if (key === '+') this.effectsManager.setEffect('grow');
            else if (key === '-') this.effectsManager.setEffect('shrink');
        });
    }

    setupMouseWheelHandler() {
        document.addEventListener('wheel', (event) => {
            if (event.ctrlKey || event.metaKey) {
                event.preventDefault();
                this.effectsManager.handlePixelationWheel(event);
            }
        }, { passive: false });
    }

    start() {
        this.streamManager.startStreaming();
        this.effectsManager.initialize();
    }
}

// Export initialization function
export function initializeApp(effects) {
    return new App(effects).initialize();
}
