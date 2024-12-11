import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import threading
import logging
from typing import Optional, Tuple, List
from ..core.app_state import ProcessingMetrics, app_state
import time
import math
import os

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the SAM model instance and predictions. This class handles model
    initialization, inference, and performance tracking for the segmentation model.
    
    The ModelManager is responsible for:
    - Loading and initializing the SAM model
    - Processing frames for prediction
    - Generating and managing segmentation masks
    - Tracking model performance metrics
    """
    
    def __init__(self):
        self.model = None
        self.predictor = None
        self.metrics = ProcessingMetrics()
        self.prediction_lock = threading.Lock()
        
        # Model configuration
        self.points_per_side = 2  # Grid points for prediction
        self.min_mask_region_area = 100  # Minimum area for valid masks
        
        # Performance tracking
        self._prediction_times = []
        self.max_timing_samples = 100
        
        # Register with application state
        app_state.model_manager = self

    def initialize(self, model_type: str, checkpoint_path: str) -> bool:
        """
        Initialize the SAM model with specified configuration.
        
        Args:
            model_type: Type of SAM model to use (e.g., 'vit_t')
            checkpoint_path: Path to model weights file
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Model checkpoint not found: {checkpoint_path}")
                app_state.set_component_status('model', False)
                return False

            logger.info(f"Loading SAM model type: {model_type}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Initialize model with specified configuration
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.model.to(device=device)
            self.model.eval()
            
            # Initialize predictor
            self.predictor = SamPredictor(self.model)
            
            logger.info("Model initialization complete")
            app_state.set_component_status('model', True)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            app_state.set_component_status('model', False)
            return False

    def predict(self, frame: np.ndarray) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """Generate predictions for a given frame"""
        try:
            if frame is None:
                logger.error("Received None frame")
                return None, None
                
            start_time = time.time()
            
            with self.prediction_lock:
                # Resize frame for efficient processing
                scaled_size = (256, 144)  # Reduce resolution for speed
                frame_small = cv2.resize(frame, scaled_size)
                
                # Ensure RGB format
                if frame_small.shape[2] == 3:
                    if frame_small[0,0,0] > frame_small[0,0,2]:  # BGR check
                        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame_small
                else:
                    logger.error(f"Unexpected frame format: {frame_small.shape}")
                    return None, None

                try:
                    # Reset predictor
                    self.predictor.reset_image()
                    # Set new image
                    self.predictor.set_image(frame_rgb)
                except Exception as e:
                    logger.error(f"Error setting predictor image: {e}")
                    return None, None

                # Generate points grid
                points = self._generate_points_grid(*frame_rgb.shape[:2])

                # Get predictions
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points)),
                    multimask_output=False  # Single mask for better performance
                )

                # Scale masks back to original size
                if masks is not None:
                    masks = [cv2.resize(mask.astype(np.uint8), frame.shape[:2][::-1]) > 0 
                            for mask in [masks[0]]]  # Only take first mask
                    scores = [scores[0]]  # Match scores to masks

                return masks, scores
                    
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None, None

    def _generate_points_grid(self, height: int, width: int) -> np.ndarray:
        """
        Generate a grid of points for prediction.
        
        Args:
            height: Frame height
            width: Frame width
            
        Returns:
            numpy.ndarray: Grid points for prediction
        """
        x = np.linspace(0, width, self.points_per_side)
        y = np.linspace(0, height, self.points_per_side)
        xv, yv = np.meshgrid(x, y)
        return np.stack([xv.flatten(), yv.flatten()], axis=1)

    def _post_process_masks(
        self, 
        masks: np.ndarray, 
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-process predicted masks to improve quality.
        
        Args:
            masks: Predicted segmentation masks
            scores: Confidence scores for masks
            
        Returns:
            tuple: (processed_masks, filtered_scores)
        """
        processed_masks = []
        filtered_scores = []
        
        for mask, score in zip(masks, scores):
            # Filter small regions
            if np.sum(mask) > self.min_mask_region_area:
                # Clean up mask edges
                cleaned_mask = cv2.morphologyEx(
                    mask.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    np.ones((3,3), np.uint8)
                )
                processed_masks.append(cleaned_mask)
                filtered_scores.append(score)
        
        return np.array(processed_masks), np.array(filtered_scores)

    def _update_metrics(
        self, 
        prediction_time: float, 
        masks: Optional[np.ndarray], 
        scores: Optional[np.ndarray]
    ) -> None:
        """
        Update performance metrics for the model.
        
        Args:
            prediction_time: Time taken for prediction
            masks: Generated masks
            scores: Confidence scores
        """
        self._prediction_times.append(prediction_time)
        if len(self._prediction_times) > self.max_timing_samples:
            self._prediction_times.pop(0)
        
        self.metrics.prediction_time = prediction_time
        self.metrics.total_masks = len(masks) if masks is not None else 0
        self.metrics.avg_confidence = np.mean(scores) if scores is not None else 0

    def _sanitize_float(self, value):
        """Sanitize a float value to ensure no NaN or Infinity."""
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 0.0  # Default to 0.0 for invalid float values
        return value

    def get_metrics(self) -> dict:
        """
        Get current model metrics.

        Returns:
            dict: Current performance metrics.
        """
        return {
            'prediction_time': self._sanitize_float(float(self.metrics.prediction_time)),
            'avg_prediction_time': self._sanitize_float(float(np.mean(self._prediction_times))),
            'total_masks': int(self.metrics.total_masks),  # Assuming total_masks is always a valid int
            'avg_confidence': self._sanitize_float(float(self.metrics.avg_confidence)),
            'device': str(next(self.model.parameters()).device)
        }
