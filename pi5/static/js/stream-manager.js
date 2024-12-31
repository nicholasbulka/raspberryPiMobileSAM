// stream-manager.js
export class StreamManager {
    constructor() {
        // Canvas elements and contexts
        this.rawCanvas = null;
        this.processedCanvas = null;
        this.rawCtx = null;
        this.processedCtx = null;

        // Stream properties
        this.streamInterval = null;
        this.updateInterval = 50; // 20 FPS default
        
        // Fullscreen properties
        this.isFullscreen = false;
        this.originalDimensions = {
            raw: { width: 640, height: 360 },
            processed: { width: 640, height: 360 }
        };
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.fpsDisplay = null;
        
        // Cursor timeout for fullscreen
        this.cursorTimeout = null;
    }

    initialize() {
        this.initializeCanvases();
        this.setupFullscreenHandlers();
        this.setupRawFeedToggle();
        this.setupFlipToggle();
        this.startPerformanceMonitoring();
        this.startStreaming();
    }

    initializeCanvases() {
        this.rawCanvas = document.getElementById('rawCanvas');
        this.processedCanvas = document.getElementById('processedCanvas');
        
        if (!this.rawCanvas || !this.processedCanvas) {
            console.error('Canvas elements not found');
            return;
        }

        this.rawCtx = this.setupCanvasContext(this.rawCanvas);
        this.processedCtx = this.setupCanvasContext(this.processedCanvas);
    }

    setupCanvasContext(canvas) {
        canvas.width = this.originalDimensions.raw.width;
        canvas.height = this.originalDimensions.raw.height;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        return ctx;
    }

    // Stream Management Methods
    startStreaming() {
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
        }
        this.streamInterval = setInterval(() => this.updateStreams(), this.updateInterval);
    }

    stopStreaming() {
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }
    }

    async updateStreams() {
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
                ctx.imageSmoothingEnabled = false;
                const destWidth = this.isFullscreen ? ctx.canvas.width : this.originalDimensions.raw.width;
                const destHeight = this.isFullscreen ? ctx.canvas.height : this.originalDimensions.raw.height;
                ctx.drawImage(img, 0, 0, destWidth, destHeight);
                resolve();
            };
            img.onerror = reject;
            img.src = 'data:image/jpeg;base64,' + base64Data;
        });
    }

    // Camera Flip Toggles
    setupFlipToggle() {
        // Horizontal flip
        const toggleButtonH = document.getElementById('toggle-flip-h');
        if (toggleButtonH) {
            toggleButtonH.addEventListener('click', async () => {
                try {
                    const response = await fetch('/flip_camera_h', { method: 'POST' });
                    const data = await response.json();
                    toggleButtonH.classList.toggle('active', data.flipped);
                } catch (error) {
                    console.error('Error toggling horizontal camera flip:', error);
                }
            });
        }

        // Vertical flip
        const toggleButtonV = document.getElementById('toggle-flip-v');
        if (toggleButtonV) {
            toggleButtonV.addEventListener('click', async () => {
                try {
                    const response = await fetch('/flip_camera_v', { method: 'POST' });
                    const data = await response.json();
                    toggleButtonV.classList.toggle('active', data.flipped);
                } catch (error) {
                    console.error('Error toggling vertical camera flip:', error);
                }
            });
        }
    }

    // Fullscreen Management Methods
    setupFullscreenHandlers() {
        const canvasContainers = document.querySelectorAll('.canvas-container');
        canvasContainers.forEach(container => {
            const canvas = container.querySelector('canvas');
            const fullscreenBtn = container.querySelector('.fullscreen-button');

            if (canvas && fullscreenBtn) {
                canvas.addEventListener('click', () => this.toggleFullscreen(container));
                fullscreenBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleFullscreen(container);
                });
            }
        });

        document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
        document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
        
        // Add escape key handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isFullscreen) {
                this.exitFullscreen();
            }
        });
    }

    async toggleFullscreen(container) {
        if (!this.isFullscreen) {
            await this.enterFullscreen(container);
        } else {
            await this.exitFullscreen();
        }
    }

    async enterFullscreen(container) {
        try {
            if (container.requestFullscreen) {
                await container.requestFullscreen();
            } else if (container.webkitRequestFullscreen) {
                await container.webkitRequestFullscreen();
            }
            
            const canvas = container.querySelector('canvas');
            this.enterFullscreenMode(container, canvas);
        } catch (err) {
            console.error('Error entering fullscreen:', err);
        }
    }

    async exitFullscreen() {
        try {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            }
        } catch (err) {
            console.error('Error exiting fullscreen:', err);
        }
    }

    enterFullscreenMode(container, canvas) {
        this.isFullscreen = true;
        
        // Hide the stream title in fullscreen
        const streamTitle = container.closest('.stream').querySelector('.stream-title');
        if (streamTitle) {
            streamTitle.style.display = 'none';
        }
        
        // Adjust container styles
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.padding = '0';
        container.style.margin = '0';
        container.style.backgroundColor = 'black';
        
        // Adjust canvas styles
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.maxWidth = 'none';
        canvas.style.maxHeight = 'none';
        canvas.style.margin = '0';
        canvas.style.padding = '0';
        canvas.style.objectFit = 'contain';
        canvas.style.borderRadius = '0';
        canvas.style.border = 'none';
        
        // Set canvas dimensions to screen size
        canvas.width = window.screen.width;
        canvas.height = window.screen.height;
        
        this.setupFullscreenCursor();
    }

    handleFullscreenChange() {
        const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
        this.isFullscreen = !!fullscreenElement;
        
        if (!this.isFullscreen) {
            this.exitFullscreenMode();
        }
    }

    exitFullscreenMode() {
        const canvases = document.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const container = canvas.closest('.stream');
            
            // Reset dimensions
            canvas.width = this.originalDimensions.raw.width;
            canvas.height = this.originalDimensions.raw.height;
            
            // Reset styles
            container.style = '';
            canvas.style = '';
            canvas.style.cursor = 'pointer';
            
            // Show the stream title again
            const streamTitle = container.querySelector('.stream-title');
            if (streamTitle) {
                streamTitle.style.display = '';
            }
            
            // Ensure smooth rendering is disabled
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
        });
        
        document.body.style.cursor = 'default';
    }

    setupFullscreenCursor() {
        document.addEventListener('mousemove', () => {
            if (this.isFullscreen) {
                document.body.style.cursor = 'default';
                clearTimeout(this.cursorTimeout);
                this.cursorTimeout = setTimeout(() => {
                    if (this.isFullscreen) {
                        document.body.style.cursor = 'none';
                    }
                }, 2000);
            }
        });
    }

    // Raw Feed Toggle
    setupRawFeedToggle() {
        const toggleButton = document.getElementById('toggle-raw');
        if (toggleButton) {
            toggleButton.addEventListener('click', () => {
                const rawStream = document.querySelector('.stream.raw');
                if (rawStream) {
                    const isHidden = rawStream.style.display === 'none' || !rawStream.style.display;
                    rawStream.style.display = isHidden ? 'block' : 'none';
                    toggleButton.classList.toggle('active', isHidden);
                    toggleButton.textContent = isHidden ? 'Hide Raw Feed' : 'Show Raw Feed';
                }
            });
        }
    }

    // Performance Monitoring
    startPerformanceMonitoring() {
        this.setupFPSDisplay();
        requestAnimationFrame(() => this.updateFPS());
    }

    setupFPSDisplay() {
        this.fpsDisplay = document.createElement('div');
        this.fpsDisplay.className = 'fps-display';
        document.body.appendChild(this.fpsDisplay);
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
        
        requestAnimationFrame(() => this.updateFPS());
    }

    // Public getters
    getIsFullscreen() {
        return this.isFullscreen;
    }

    getStreamStatus() {
        return !!this.streamInterval;
    }
}

export default StreamManager;
