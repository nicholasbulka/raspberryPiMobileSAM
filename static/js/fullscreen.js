class FullscreenManager {
    constructor() {
        this.isFullscreen = false;
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
    }

    setupEscapeHandler() {
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.isFullscreen) {
                const fullscreenElement = document.querySelector('.fullscreen');
                if (fullscreenElement) {
                    this.toggleFullscreen(fullscreenElement);
                }
            }
        });
    }

    toggleFullscreen(element) {
        const controls = document.querySelector('.controls');
        const container = element.closest('.stream');
        
        if (!this.isFullscreen) {
            container.classList.add('fullscreen');
            controls.classList.add('hidden');
            this.isFullscreen = true;
            
            // Adjust canvas size for fullscreen
            const canvas = container.querySelector('canvas');
            this.originalDimensions = {
                width: canvas.width,
                height: canvas.height
            };
            
            // Maintain aspect ratio in fullscreen
            const windowRatio = window.innerWidth / window.innerHeight;
            const canvasRatio = canvas.width / canvas.height;
            
            if (windowRatio > canvasRatio) {
                canvas.style.height = '100vh';
                canvas.style.width = 'auto';
            } else {
                canvas.style.width = '100vw';
                canvas.style.height = 'auto';
            }
        } else {
            container.classList.remove('fullscreen');
            controls.classList.remove('hidden');
            this.isFullscreen = false;
            
            // Restore original canvas dimensions
            const canvas = container.querySelector('canvas');
            if (this.originalDimensions) {
                canvas.width = this.originalDimensions.width;
                canvas.height = this.originalDimensions.height;
                canvas.style.width = '';
                canvas.style.height = '';
            }
        }
    }
}

export default FullscreenManager;
