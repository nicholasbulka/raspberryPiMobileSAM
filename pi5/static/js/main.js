// main.js
class VideoStreamApp {
    constructor() {
        // Will be initialized later
        this.effectsManager = null;
        this.streamManager = null;
    }

    async initialize() {
        try {
            // Import managers dynamically
            const [{ default: EffectsManager }, { default: StreamManager }] = await Promise.all([
                import('./effects.js'),
                import('./stream-manager.js')
            ]);
            
            // Initialize stream manager
            this.streamManager = new StreamManager();
            this.streamManager.initialize();
            
            // Initialize effects manager
            this.effectsManager = new EffectsManager();
            this.effectsManager.initialize({
                onEffectChange: (effect) => this.handleEffectChange(effect),
                onParamChange: (params) => this.handleParamChange(params),
                effectsList: window.EFFECTS_LIST
            });
            
        } catch (error) {
            console.error('Error initializing app:', error);
        }
    }

    handleEffectChange(effect) {
        console.log('Effect changed:', effect);
        // Additional effect change handling if needed
    }

    handleParamChange(params) {
        console.log('Parameters updated:', params);
        // Additional parameter change handling if needed
    }
}

// Initialize the application when the DOM is ready
const app = new VideoStreamApp();
document.addEventListener('DOMContentLoaded', () => {
    app.initialize();
});

// Export the app instance
window.app = app;
