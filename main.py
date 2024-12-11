import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import time
import os
import io
import base64
from flask import Flask, render_template, jsonify, request
import threading
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
import queue
import signal
import sys
import psutil
from effects_manager import effects_manager

# Configure logging with timestamp, logger name, and level for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """
    Tracks various performance metrics for the application.
    These metrics help monitor system health and performance.
    """
    fps: float = 0.0  # Frames processed per second
    prediction_time: float = 0.0  # Time taken for model predictions
    effect_time: float = 0.0  # Time taken to apply effects
    total_masks: int = 0  # Number of masks detected
    avg_confidence: float = 0.0  # Average confidence score of detections

class ApplicationState:
    """
    Maintains global application state and component references.
    This centralized state management ensures proper component access across threads.
    """
    def __init__(self):
        self.camera_manager = None  # Handles camera operations
        self.model_manager = None   # Manages the SAM model
        self.frame_processor = None # Processes frames with effects
        self.web_server = None      # Handles HTTP requests
        self.shutdown_event = threading.Event()
        self.metrics = ProcessingMetrics()

# Initialize global application state and Flask app
app_state = ApplicationState()
app = Flask(__name__)

class CameraManager:
    """
    Manages camera operations including frame capture and basic processing.
    Implements thread-safe frame access and performance monitoring.
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer recent frames
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.metrics = ProcessingMetrics()
        self._frame_count = 0
        self._start_time = time.time()

    def initialize(self) -> bool:
        """
        Initialize camera device with error handling.
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
                
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False

    def start(self) -> None:
        """Start frame capture thread"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Camera capture thread started")

    def stop(self) -> None:
        """Stop frame capture and release resources"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if self.capture:
            self.capture.release()
        logger.info("Camera resources released")

    def _capture_frames(self) -> None:
        """
        Continuous frame capture loop with performance monitoring.
        Handles frame acquisition, basic processing, and metrics tracking.
        """
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    time.sleep(0.1)
                    continue

                # Flip frame based on camera orientation
                frame = cv2.flip(frame, -1)

                # Thread-safe frame update
                with self.frame_lock:
                    self.current_frame = frame

                # Update performance metrics
                self._frame_count += 1
                elapsed_time = time.time() - self._start_time
                if elapsed_time >= 1.0:  # Update metrics every second
                    self.metrics.fps = self._frame_count / elapsed_time
                    self._frame_count = 0
                    self._start_time = time.time()

            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                time.sleep(0.1)

class ModelManager:
    """
    Manages the SAM model instance and predictions.
    Handles model initialization, inference, and performance tracking.
    """
    def __init__(self):
        self.model = None
        self.predictor = None
        self.metrics = ProcessingMetrics()
        self.prediction_lock = threading.Lock()

    def initialize(self, model_type: str, checkpoint_path: str) -> bool:
        """
        Initialize the SAM model with the specified configuration.
        
        Args:
            model_type: Type of SAM model to use
            checkpoint_path: Path to model weights
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Model checkpoint not found: {checkpoint_path}")
                return False

            logger.info("Loading SAM model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Initialize model with specified configuration
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.model.to(device=device)
            self.model.eval()
            
            self.predictor = SamPredictor(self.model)
            logger.info("Model initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False

    def predict(self, frame: np.ndarray) -> tuple:
        """
        Generate predictions with performance tracking.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (masks, scores) or (None, None) on error
        """
        try:
            start_time = time.time()
            
            with self.prediction_lock:
                # Resize frame for efficient processing
                frame_small = cv2.resize(frame, (128, 72))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                
                self.predictor.set_image(frame_rgb)
                
                # Generate points grid for prediction
                h, w = frame_small.shape[:2]
                points_per_side = 2
                x = np.linspace(0, w, points_per_side)
                y = np.linspace(0, h, points_per_side)
                xv, yv = np.meshgrid(x, y)
                points = np.stack([xv.flatten(), yv.flatten()], axis=1)

                # Generate predictions
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points)),
                    multimask_output=True
                )

                # Update performance metrics
                self.metrics.prediction_time = time.time() - start_time
                self.metrics.total_masks = len(masks) if masks is not None else 0
                self.metrics.avg_confidence = np.mean(scores) if scores is not None else 0

                return masks, scores
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None, None

class FrameProcessor:
    """
    Handles frame processing pipeline including predictions and effects.
    Manages the main processing loop and coordinates between components.
    """
    def __init__(self, model_manager: ModelManager, effects_manager):
        self.model_manager = model_manager
        self.effects_manager = effects_manager
        self.processed_frame = None
        self.processing_lock = threading.Lock()
        self.is_running = False
        self.metrics = ProcessingMetrics()

    def start(self) -> None:
        """Start the processing loop"""
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        logger.info("Frame processor started")

    def stop(self) -> None:
        """Stop processing and clean up"""
        self.is_running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        logger.info("Frame processor stopped")

    def _process_loop(self) -> None:
        """
        Main processing loop handling frame processing and effect application.
        Implements error handling and performance monitoring.
        """
        while self.is_running:
            try:
                # Get latest frame
                with app_state.camera_manager.frame_lock:
                    if app_state.camera_manager.current_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = app_state.camera_manager.current_frame.copy()

                # Generate predictions
                start_time = time.time()
                masks, scores = self.model_manager.predict(frame)

                # Apply effects
                processed = self.effects_manager.process_frame(frame, masks)
                self.metrics.effect_time = time.time() - start_time

                # Update processed frame
                with self.processing_lock:
                    self.processed_frame = processed

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

class WebServer:
    """
    Handles web server operations and client communication.
    Implements routes and handles client requests.
    """
    def __init__(self, app, app_state: ApplicationState):
        self.app = app
        self.app_state = app_state
        self.setup_routes()

    def setup_routes(self) -> None:
        """Configure Flask routes with error handling"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html', 
                                effects=self.effects_manager.get_effects_list())

        @self.app.route('/stream')
        def stream():
            try:
                with self.app_state.camera_manager.frame_lock:
                    if self.app_state.camera_manager.current_frame is None:
                        return jsonify({"error": "No frames available"}), 404

                    raw_b64 = self.encode_frame(
                        self.app_state.camera_manager.current_frame)
                    processed_b64 = self.encode_frame(
                        self.app_state.frame_processor.processed_frame)

                    return jsonify({
                        "raw": raw_b64,
                        "processed": processed_b64
                    })
            except Exception as e:
                logger.error(f"Error in stream endpoint: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/effect/<effect_name>', methods=['POST'])
        def set_effect(effect_name):
            try:
                self.effects_manager.set_effect(effect_name)
                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error setting effect: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/effect_params', methods=['POST'])
        def update_effect_params():
            try:
                self.effects_manager.update_params(request.json)
                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/debug_info')
        def get_debug_info():
            try:
                debug_info = {
                    'effects': self.effects_manager.get_debug_info(),
                    'camera': vars(self.app_state.camera_manager.metrics),
                    'model': vars(self.app_state.model_manager.metrics),
                    'system': self.get_system_metrics()
                }
                return jsonify(debug_info)
            except Exception as e:
                logger.error(f"Error getting debug info: {e}")
                return jsonify({"error": str(e)}), 500

    @staticmethod
    def encode_frame(frame: np.ndarray) -> Optional[str]:
        """
        Encode frame to base64 for transmission.
        
        Args:
            frame: Input frame to encode
            
        Returns:
            str: Base64 encoded frame or None on error
        """
        if frame is None:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """
        Get system performance metrics.
        
        Returns:
            dict: System metrics including CPU and memory usage
        """
        try:
            # Get CPU temperature (Raspberry Pi specific)
            cpu_temp = 0
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000
            except:
                pass

            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_metrics = {
                'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
                'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
                'percent': process.memory_percent()
            }

            return {
                'cpu_temp': cpu_temp,
                'memory': memory_metrics,
                'cpu_percent': psutil.cpu_percent()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

def initialize_application() -> bool:
    """
    Initialize all application components.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing application components...")

        # Initialize camera
        app_state.camera_manager = CameraManager()
        if not app_state.camera_manager.initialize():
            return False

        # Initialize model
        app_state.model_manager = ModelManager()
        model_path = os.path.join("weights", "mobile_sam.pt")
        if not app_state.model_manager.initialize("vit_t", model_path):
            return False

        # Initialize processor and web server
        app_state.frame_processor = FrameProcessor(
            app_state.model_manager, 
            effects_manager
        )
        app_state.web_server = WebServer(app, app_state)

        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info("Application initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False

def handle_shutdown(signum, frame) -> None:
    """
    Handle shutdown signals gracefully.
    Ensures proper cleanup of resources on application termination.
    
    Args:
        signum: Signal number received
        frame: Current stack frame
    """
    logger.info(f"Received shutdown signal: {signum}")
    shutdown_application()

def shutdown_application() -> None:
    """
    Clean shutdown of all application components.
    Ensures resources are properly released and threads are terminated.
    """
    logger.info("Initiating shutdown sequence...")
    
    app_state.shutdown_event.set()

    # Stop components in reverse order of initialization
    if app_state.frame_processor:
        logger.info("Stopping frame processor...")
        app_state.frame_processor.stop()

    if app_state.camera_manager:
        logger.info("Stopping camera manager...")
        app_state.camera_manager.stop()

    logger.info("Shutdown complete")
    sys.exit(0)

def check_system_requirements() -> bool:
    """
    Check if system meets minimum requirements for running the application.
    
    Returns:
        bool: True if system meets requirements, False otherwise
    """
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False

        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 1 * 1024 * 1024 * 1024:  # 1GB
            logger.error("Insufficient available memory (minimum 1GB required)")
            return False

        # Check OpenCV version
        opencv_version = cv2.__version__.split('.')
        if int(opencv_version[0]) < 4:
            logger.error("OpenCV 4.0 or higher is required")
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking system requirements: {e}")
        return False

def main():
    """
    Application entry point with comprehensive error handling.
    Coordinates initialization, execution, and shutdown of all components.
    """
    try:
        logger.info("Starting MobileSAM Application")
        
        # Check environment and requirements
        if not check_system_requirements():
            logger.error("System requirements not met")
            return

        logger.info(f"Python version: {sys.version}")
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"Torch version: {torch.__version__}")
        
        # Create necessary directories
        os.makedirs("weights", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Check model file
        model_path = os.path.join("weights", "mobile_sam.pt")
        if not os.path.exists(model_path):
            logger.error("Model file not found. Please download MobileSAM weights.")
            logger.info("wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -P weights/")
            return

        # Initialize application
        if not initialize_application():
            logger.error("Failed to initialize application")
            sys.exit(1)

        # Start components
        logger.info("Starting application components...")
        app_state.camera_manager.start()
        app_state.frame_processor.start()

        # Start web server
        logger.info("Starting web server...")
        logger.info("Server will be available at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown_application()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
