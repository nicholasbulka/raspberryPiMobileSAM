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
        this.isStreaming = false;
        this.retryCount = 0;
        this.maxRetries = 3;
        this.retryDelay = 1000;
        this.targetFPS = 15;
        this.frameInterval = 1000 / this.targetFPS;
        this.lastFrameRequest = 0;
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
            return false;
        }

        // Create FPS display
        this.fpsDisplay = this.createFPSDisplay();
        
        return true;
    }

    setupCanvas(canvas) {
        // Set canvas dimensions
        canvas.width = 640;
        canvas.height = 360;
        
        // Get and configure context
        const ctx = canvas.getContext('2d', { 
            alpha: false,
            desynchronized: true,
            willReadFrequently: false
        });
        
        if (ctx) {
            // Disable image smoothing for better performance
            ctx.imageSmoothingEnabled = false;
            ctx.webkitImageSmoothingEnabled = false;
            ctx.mozImageSmoothingEnabled = false;
            ctx.msImageSmoothingEnabled = false;
        }
        return ctx;
    }

    createFPSDisplay() {
        const display = document.createElement('div');
        display.className = 'fps-display';
        document.body.appendChild(display);
        return display;
    }

    async updateStreams() {
        const now = performance.now();
        const elapsed = now - this.lastFrameRequest;

        if (elapsed < this.frameInterval) {
            return;  // Skip frame to maintain target FPS
        }

        this.lastFrameRequest = now;

        if (!this.rawCtx || !this.processedCtx || !this.isStreaming) {
            return;
        }

        try {
            const response = await fetch('/stream', {
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Stream response not ok: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (data.raw && data.processed) {
                await Promise.all([
                    this.updateCanvas(this.rawCtx, data.raw),
                    this.updateCanvas(this.processedCtx, data.processed)
                ]);
                this.retryCount = 0; // Reset retry count on success
                this.updateFPS();
            }
        } catch (error) {
            console.error('Error updating streams:', error);
            this.handleStreamError();
        }
    }

    async updateCanvas(ctx, base64Data) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => {
                // Use createImageBitmap for better performance when available
                if (window.createImageBitmap) {
                    createImageBitmap(img).then(bitmap => {
                        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                        ctx.drawImage(bitmap, 0, 0, ctx.canvas.width, ctx.canvas.height);
                        bitmap.close();
                        resolve();
                    }).catch(err => {
                        // Fallback to direct drawing if createImageBitmap fails
                        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                        ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
                        resolve();
                    });
                } else {
                    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                    ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
                    resolve();
                }
            };
            
            img.onerror = (error) => {
                console.error('Error loading image:', error);
                reject(error);
            };

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

    handleStreamError() {
        this.retryCount++;
        if (this.retryCount <= this.maxRetries) {
            console.log(`Retrying stream connection (${this.retryCount}/${this.maxRetries})...`);
            setTimeout(() => {
                if (this.isStreaming) {
                    this.updateStreams();
                }
            }, this.retryDelay);
        } else {
            console.error('Max retry attempts reached. Stream connection failed.');
            this.stopStreaming();
            this.showStreamError();
        }
    }

    showStreamError() {
        const message = 'Stream connection lost. Please refresh the page.';
        [this.rawCtx, this.processedCtx].forEach(ctx => {
            if (ctx) {
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2);
            }
        });
    }

    startStreaming() {
        if (this.streamInterval) {
            this.stopStreaming();
        }

        this.isStreaming = true;
        this.retryCount = 0;
        this.lastFrameRequest = 0;
        
        // Initial update
        this.updateStreams();
        
        // Start regular updates with RAF for better performance
        const streamLoop = () => {
            if (this.isStreaming) {
                this.updateStreams();
                requestAnimationFrame(streamLoop);
            }
        };
        requestAnimationFrame(streamLoop);
    }

    stopStreaming() {
        this.isStreaming = false;
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }
    }

    cleanup() {
        this.stopStreaming();
        if (this.fpsDisplay && this.fpsDisplay.parentNode) {
            this.fpsDisplay.parentNode.removeChild(this.fpsDisplay);
        }
        
        // Clear canvases
        if (this.rawCtx) {
            this.rawCtx.clearRect(0, 0, this.rawCanvas.width, this.rawCanvas.height);
        }
        if (this.processedCtx) {
            this.processedCtx.clearRect(0, 0, this.processedCanvas.width, this.processedCanvas.height);
        }
    }
}