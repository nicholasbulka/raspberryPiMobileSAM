import threading
import time
import logging
import numpy as np
from typing import Optional
from ..core.app_state import ProcessingMetrics, app_state
from ..model.model_manager import ModelManager
from ..effects.effects_parameters import EffectParameters

logger = logging.getLogger(__name__)

class FrameProcessor:
    """
    Handles the main frame processing pipeline including predictions and effects.
    Coordinates between model predictions and effect application while maintaining
    performance metrics and thread safety.
    
    This class is responsible for:
    - Managing the processing pipeline
    - Coordinating between model and effects
    - Maintaining processed frame buffer
    - Tracking processing performance
    """
    
    def __init__(self, model_manager: ModelManager, effects_manager):
        """
        Initialize frame processor with required components.
        
        Args:
            model_manager: Instance of ModelManager for predictions
            effects_manager: Instance of EffectsManager for applying effects
        """
        self.model_manager = model_manager
        self.effects_manager = effects_manager
        self.processed_frame = None
        self.processing_lock = threading.Lock()
        self.is_running = False
        self.metrics = ProcessingMetrics()
        
        # Processing state
        self._last_process_time = 0
        self._frame_count = 0
        self._start_time = time.time()
        
        # Performance monitoring
        self._process_times = []
        self.max_timing_samples = 100
        
        # Register with application state
        app_state.frame_processor = self

    def start(self) -> None:
        """Start the processing loop in a separate thread"""
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        logger.info("Frame processor started")

    def stop(self) -> None:
        """Stop processing and clean up resources"""
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

                # Process frame
                processed = self._process_frame(frame)
                
                # Update processed frame buffer
                if processed is not None:
                    with self.processing_lock:
                        self.processed_frame = processed

                # Control processing rate
                self._maintain_processing_rate()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame to process
                
        Returns:
            Processed frame or None on error
        """
        try:
            start_time = time.time()

            # Generate predictions
            masks, scores = self.model_manager.predict(frame)
            
            if masks is not None and len(masks) > 0:
                processed = self.effects_manager.process_frame(frame, masks)
                process_time = time.time() - start_time
                self._update_metrics(process_time)
                return processed
                
            # If no masks, return original frame
            return frame.copy()

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame.copy()

    def _update_metrics(self, process_time: float) -> None:
        """
        Update processing performance metrics.
        
        Args:
            process_time: Time taken to process frame
        """
        self._process_times.append(process_time)
        if len(self._process_times) > self.max_timing_samples:
            self._process_times.pop(0)
        
        self._frame_count += 1
        elapsed = time.time() - self._start_time
        
        if elapsed >= 1.0:
            self.metrics.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()

    def _maintain_processing_rate(self) -> None:
        """Control processing rate to maintain stable performance"""
        target_interval = 1.0 / 30  # Target 30 FPS
        elapsed = time.time() - self._last_process_time
        
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        self._last_process_time = time.time()

    def get_metrics(self) -> dict:
        """
        Get current processing metrics.
        
        Returns:
            dict: Current performance metrics
        """
        return {
            'fps': self.metrics.fps,
            'avg_process_time': np.mean(self._process_times) if self._process_times else 0,
            'min_process_time': min(self._process_times) if self._process_times else 0,
            'max_process_time': max(self._process_times) if self._process_times else 0,
            'total_frames_processed': self._frame_count
        }
