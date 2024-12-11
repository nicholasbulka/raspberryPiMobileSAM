import os
import sys
import signal
import threading
from flask import Flask
import logging
from typing import Optional

# Use absolute imports
from sam_app.core.app_state import app_state
from sam_app.camera.camera_manager import CameraManager
from sam_app.model.model_manager import ModelManager
from sam_app.processing.frame_processor import FrameProcessor
from sam_app.web.web_server import WebServer
from sam_app.effects.effects_parameters import EffectParameters
from sam_app.effects.effects_manager import EffectsManager  # Added import
from sam_app.utils.logging_config import logging_config
from sam_app.utils.system_monitor import system_monitor


# Initialize logging
logging_config.configure(app_name="sam_app")
logger = logging.getLogger(__name__)

class Application:
    """
    Main application class coordinating all components.
    Handles initialization, lifecycle management, and shutdown procedures.
    """
    
    def __init__(self):
        self.flask_app = Flask(__name__)
        self.stop_event = threading.Event()

    def initialize(self) -> bool:
        """
        Initialize all application components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing application...")
            
            # Create necessary directories
            os.makedirs("weights", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Check model file
            model_path = os.path.join("weights", "mobile_sam.pt")
            if not os.path.exists(model_path):
                logger.error(
                    "Model file not found. Please download MobileSAM weights:\n"
                    "wget https://github.com/ChaoningZhang/MobileSAM/raw/"
                    "master/weights/mobile_sam.pt -P weights/"
                )
                return False

            # Initialize components in correct order
            self.init_camera()
            self.init_model(model_path)
            self.init_effects()  # Added effects initialization
            self.init_processor()
            self.init_web_server()
            
            # Start system monitoring
            system_monitor.start()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
            
            logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            return False

    def init_camera(self) -> None:
        """Initialize camera component"""
        logger.info("Initializing camera...")
        app_state.camera_manager = CameraManager()
        if not app_state.camera_manager.initialize():
            raise RuntimeError("Camera initialization failed")

    def init_model(self, model_path: str) -> None:
        """
        Initialize ML model component.
        
        Args:
            model_path: Path to model weights
        """
        logger.info("Initializing model...")
        app_state.model_manager = ModelManager()
        if not app_state.model_manager.initialize("vit_t", model_path):
            raise RuntimeError("Model initialization failed")

    def init_effects(self) -> None:
        """Initialize effects manager component"""
        logger.info("Initializing effects manager...")
        app_state.effects_manager = EffectsManager()
        if not app_state.effects_manager.initialize():
            raise RuntimeError("Effects manager initialization failed")

    def init_processor(self) -> None:
        """Initialize frame processor component"""
        logger.info("Initializing frame processor...")
        app_state.frame_processor = FrameProcessor(
            app_state.model_manager,
            app_state.effects_manager
        )

    def init_web_server(self) -> None:
        """Initialize web server component"""
        logger.info("Initializing web server...")
        app_state.web_server = WebServer(self.flask_app, app_state)

    def start(self) -> None:
        """Start all application components"""
        try:
            logger.info("Starting application components...")
            
            # Start components
            app_state.camera_manager.start()
            app_state.frame_processor.start()
            
            # Start web server
            logger.info("Starting web server...")
            self.flask_app.run(
                host='0.0.0.0',
                port=5000,
                threaded=True,
                use_reloader=False
            )
            
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            self.shutdown()

    def shutdown(self, signum: Optional[int] = None, frame=None) -> None:
        """
        Clean shutdown of all components.
        
        Args:
            signum: Signal number (if called as signal handler)
            frame: Stack frame (if called as signal handler)
        """
        logger.info("Initiating shutdown sequence...")
        
        # Set stop event
        self.stop_event.set()
        
        # Stop components in reverse order
        if app_state.frame_processor:
            logger.info("Stopping frame processor...")
            app_state.frame_processor.stop()
            
        if app_state.camera_manager:
            logger.info("Stopping camera manager...")
            app_state.camera_manager.stop()
            
        # Stop system monitoring
        logger.info("Stopping system monitor...")
        system_monitor.stop()
        
        logger.info("Shutdown complete")
        sys.exit(0)

    def handle_shutdown(self, signum: int, frame) -> None:
        """Signal handler for shutdown signals"""
        logger.info(f"Received shutdown signal: {signum}")
        self.shutdown(signum, frame)

def main():
    """Application entry point"""
    try:
        # Create and initialize application
        app = Application()
        
        if not app.initialize():
            logger.error("Application initialization failed")
            sys.exit(1)
            
        # Start application
        app.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        app.shutdown()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()