export default class StreamManager {
    constructor() {
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.fpsDisplay = null;
        this.streamInterval = null;
        this.rawCanvas = null;
        this.processedCanvas = null;
        this.rawCtx = null;
        this.processedCtx = null;
        
        this.initialize();
    }

    initialize() {
        // Initialize canvases
        this.rawCanvas = document.getElementById('rawCanvas');
        this.processedCanvas = document.getElementById('processedCanvas');
        
        if (this.rawCanvas && this.processedCanvas) {
            this.rawCtx = this.setupCanvas(this.rawCanvas);
            this.processedCtx = this.setupCanvas(this.processedCanvas);
        } else {
            console.error('Canvas elements not found');
        }

        // Create FPS display
        this.fpsDisplay = this.createFPSDisplay();
        
        // Start animation loop
        this.animate();
    }

    setupCanvas(canvas) {
        canvas.width = 640;
        canvas.height = 360;
        return canvas.getContext('2d');
    }

    createFPSDisplay() {
        const display = document.createElement('div');
        display.className = 'fps-display';
        document.body.appendChild(display);
        return display;
    }

    async updateStreams() {
        if (!this.rawCtx || !this.processedCtx) return;

        try {
            const response = await fetch('/stream');
            if (!response.ok) {
                console.error('Stream response not ok:', response.status);
                return;
            }
            
            const data = await response.json();
            
            if (data.raw && data.processed) {
                await Promise.all([
                    this.updateCanvas(this.rawCtx, data.raw),
                    this.updateCanvas(this.processedCtx, data.processed)
                ]);
            }
        } catch (error) {
            console.error('Error updating streams:', error);
        }
    }

    async updateCanvas(ctx, base64Data) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                resolve();
            };
            img.onerror = reject;
            img.src = 'data:image/jpeg;base64,' + base64Data;
        });
    }

    updateFPS() {
        this.frameCount++;
        const currentTime = performance.now();
        const elapsed = currentTime - this.lastFrameTime;
        
        if (elapsed >= 1000) {
            const fps = (this.frameCount / elapsed) * 1000;
            if (this.fpsDisplay) {
                this.fpsDisplay.textContent = `FPS: ${fps.toFixed(1)}`;
            }
            this.frameCount = 0;
            this.lastFrameTime = currentTime;
        }
    }

    animate() {
        this.updateFPS();
        requestAnimationFrame(() => this.animate());
    }

    startStreaming() {
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
        }
        this.streamInterval = setInterval(() => this.updateStreams(), 33); // ~30 FPS
    }

    stopStreaming() {
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }
    }
}
