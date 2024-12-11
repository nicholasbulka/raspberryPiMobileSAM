import cv2
import numpy as np
import threading
import time
import queue
import logging
from typing import Optional, Tuple

# Use absolute imports
from sam_app.core.app_state import ProcessingMetrics, app_state



# Configure logger for camera module
logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages camera operations including frame capture and basic processing.
    Provides thread-safe frame access and performance monitoring.
    
    This class handles:
    - Camera initialization and configuration
    - Continuous frame capture
    - Frame preprocessing
    - Performance metrics tracking
    - Resource management
    
    Attributes:
        camera_index: Index of the camera device
        capture: OpenCV VideoCapture instance
        frame_queue: Buffer for recent frames
        current_frame: Most recent captured frame
        metrics: Performance tracking metrics
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera manager with specified device
        
        Args:
            camera_index: Index of the camera device to use
        """
        self.camera_index = camera_index
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.metrics = ProcessingMetrics()
        
        # Frame processing stats
        self._frame_count = 0
        self._start_time = time.time()
        self._last_frame_time = 0
        
        # Camera settings
        self._frame_width = 640
        self._frame_height = 360
        self._target_fps = 30
        
        # Register with application state
        app_state.camera_manager = self

    def initialize(self) -> bool:
        """
        Initialize camera device with error handling
        
        Returns:
            bool: True if initialization successful, False otherwise
        
        This method:
        1. Opens the camera device
        2. Sets resolution and frame rate
        3. Verifies camera accessibility
        4. Updates component status
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            # Configure camera parameters
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self.capture.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                app_state.set_component_status('camera', False)
                return False
            
            # Verify camera settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            app_state.set_component_status('camera', True)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            app_state.set_component_status('camera', False)
            return False

    def start(self) -> None:
        """
        Start frame capture thread
        
        Initializes and starts the background thread for continuous
        frame capture while monitoring performance.
        """
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Camera capture thread started")

    def stop(self) -> None:
        """
        Stop frame capture and release resources
        
        Ensures clean shutdown of camera resources and threads.
        """
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if self.capture:
            self.capture.release()
        logger.info("Camera resources released")

    def _capture_frames(self) -> None:
        """
        Continuous frame capture loop with performance monitoring
        
        Handles:
        - Frame acquisition
        - Basic preprocessing
        - Performance metrics tracking
        - Error recovery
        """
        frame_drop_count = 0
        
        while self.is_running:
            try:
                # Calculate time since last frame
                current_time = time.time()
                frame_delay = current_time - self._last_frame_time
                
                # Check if we're meeting target FPS
                if frame_delay < 1.0 / self._target_fps:
                    continue
                
                ret, frame = self.capture.read()
                if not ret:
                    frame_drop_count += 1
                    if frame_drop_count > 10:
                        logger.error("Multiple frame drops detected")
                        self._attempt_camera_recovery()
                    time.sleep(0.1)
                    continue

                frame_drop_count = 0
                self._last_frame_time = current_time

                # Basic frame preprocessing
                frame = self._preprocess_frame(frame)

                # Thread-safe frame update
                with self.frame_lock:
                    self.current_frame = frame

                # Update performance metrics
                self._update_metrics()

            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                time.sleep(0.1)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing to captured frame
        
        Args:
            frame: Raw camera frame
            
        Returns:
            Preprocessed frame
        """
        # Flip frame based on camera orientation
        frame = cv2.flip(frame, -1)
        
        # Basic color correction and normalization could be added here
        
        return frame

    def _update_metrics(self) -> None:
        """Update performance metrics"""
        self._frame_count += 1
        elapsed_time = time.time() - self._start_time
        
        if elapsed_time >= 1.0:  # Update metrics every second
            self.metrics.fps = self._frame_count / elapsed_time
            self._frame_count = 0
            self._start_time = time.time()

    def _attempt_camera_recovery(self) -> None:
        """
        Attempt to recover from camera errors
        
        Tries to:
        1. Release and reacquire camera
        2. Reset camera parameters
        3. Verify camera operation
        """
        logger.info("Attempting camera recovery...")
        try:
            self.capture.release()
            time.sleep(1.0)
            self.capture = cv2.VideoCapture(self.camera_index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self.capture.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            if self.capture.isOpened():
                logger.info("Camera recovery successful")
            else:
                logger.error("Camera recovery failed")
                
        except Exception as e:
            logger.error(f"Error during camera recovery: {e}")

    def get_metrics(self) -> dict:
        """
        Get current camera metrics
        
        Returns:
            Dictionary of current performance metrics
        """
        return {
            'fps': self.metrics.fps,
            'frame_drops': self._frame_count,
            'resolution': f"{self._frame_width}x{self._frame_height}",
            'target_fps': self._target_fps
        }
