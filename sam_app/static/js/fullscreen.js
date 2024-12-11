export default class FullscreenManager {
    constructor() {
        this.isFullscreen = false;
        this.originalDimensions = {};
        this.setupFullscreenHandlers();
        this.setupEscapeHandler();
    }

    setupFullscreenHandlers() {
        const canvasContainers = document.querySelectorAll('.canvas-container');
        
        canvasContainers.forEach(container => {
            const canvas = container.querySelector('canvas');
            const fullscreenBtn = container.querySelector('.fullscreen-button');

            // Canvas click handler
            canvas.addEventListener('click', () => {
                this.toggleFullscreen(container);
            });

            // Button click handler
            fullscreenBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleFullscreen(container);
            });
        });

        // Handle fullscreen change events
        document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
        document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
    }

    setupEscapeHandler() {
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.isFullscreen) {
                this.exitFullscreen();
            }
        });
    }

    async toggleFullscreen(element) {
        const container = element.closest('.stream');
        const canvas = container.querySelector('canvas');
        const controls = document.querySelector('.controls');
        const debugPanel = document.getElementById('debug-panel');
        
        if (!this.isFullscreen) {
            await this.enterFullscreen(container, canvas, controls, debugPanel);
        } else {
            await this.exitFullscreen();
        }
    }

    async enterFullscreen(container, canvas, controls, debugPanel) {
        try {
            // Store original dimensions
            this.originalDimensions = {
                width: canvas.width,
                height: canvas.height,
                style: {
                    container: container.style.cssText,
                    canvas: canvas.style.cssText
                }
            };

            // Enter fullscreen
            if (container.requestFullscreen) {
                await container.requestFullscreen();
            } else if (container.webkitRequestFullscreen) {
                await container.webkitRequestFullscreen();
            }

            // Configure fullscreen styles
            this.applyFullscreenStyles(container, canvas, controls, debugPanel);
            this.isFullscreen = true;

        } catch (error) {
            console.error('Error entering fullscreen:', error);
        }
    }

    async exitFullscreen() {
        try {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            }

            this.isFullscreen = false;

        } catch (error) {
            console.error('Error exiting fullscreen:', error);
        }
    }

    applyFullscreenStyles(container, canvas, controls, debugPanel) {
        // Configure container
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.padding = '0';
        container.style.margin = '0';
        container.style.backgroundColor = 'black';
        container.style.position = 'relative';

        // Configure canvas
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.maxWidth = 'none';
        canvas.style.maxHeight = 'none';
        canvas.style.margin = '0';
        canvas.style.objectFit = 'contain';
        canvas.style.backgroundColor = 'black';

        // Hide controls and debug panel
        if (controls) controls.style.display = 'none';
        if (debugPanel) debugPanel.style.display = 'none';

        // Hide cursor after delay
        this.setupCursorAutoHide();
    }

    handleFullscreenChange() {
        if (!document.fullscreenElement && !document.webkitFullscreenElement) {
            this.restoreOriginalStyles();
        }
    }

    restoreOriginalStyles() {
        const container = document.querySelector('.stream.fullscreen');
        if (!container) return;

        const canvas = container.querySelector('canvas');
        const controls = document.querySelector('.controls');
        const debugPanel = document.getElementById('debug-panel');

        // Restore container styles
        container.style.cssText = this.originalDimensions.style.container;
        container.classList.remove('fullscreen');

        // Restore canvas dimensions and styles
        canvas.width = this.originalDimensions.width;
        canvas.height = this.originalDimensions.height;
        canvas.style.cssText = this.originalDimensions.style.canvas;

        // Show controls and debug panel
        if (controls) controls.style.display = '';
        if (debugPanel) debugPanel.style.display = '';

        // Reset cursor
        document.body.style.cursor = '';
        this.isFullscreen = false;
    }

    setupCursorAutoHide() {
        let cursorTimeout;
        const container = document.querySelector('.stream.fullscreen');
        
        if (!container) return;

        container.addEventListener('mousemove', () => {
            container.style.cursor = 'default';
            clearTimeout(cursorTimeout);
            
            cursorTimeout = setTimeout(() => {
                if (this.isFullscreen) {
                    container.style.cursor = 'none';
                }
            }, 2000);
        });
    }

    handleEscape() {
        if (this.isFullscreen) {
            this.exitFullscreen();
        }
    }
}