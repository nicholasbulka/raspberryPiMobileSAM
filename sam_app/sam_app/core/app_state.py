from dataclasses import dataclass
import threading
from typing import Optional

@dataclass
class ProcessingMetrics:
    """
    Tracks various performance metrics for the application components.
    Provides a centralized way to monitor system health and performance.
    
    Attributes:
        fps: Frames processed per second
        prediction_time: Time taken for model predictions
        effect_time: Time taken to apply effects
        total_masks: Number of masks detected
        avg_confidence: Average confidence score of detections
    """
    fps: float = 0.0
    prediction_time: float = 0.0
    effect_time: float = 0.0
    total_masks: int = 0
    avg_confidence: float = 0.0
    
    def to_dict(self):
        """Convert metrics to dictionary for serialization"""
        return {
            'fps': self.fps,
            'prediction_time': self.prediction_time,
            'effect_time': self.effect_time,
            'total_masks': self.total_masks,
            'avg_confidence': self.avg_confidence
        }

class ApplicationState:
    """
    Maintains global application state and component references.
    Acts as a central point for component access and state management.
    
    This class follows the singleton pattern to ensure only one instance
    exists throughout the application lifecycle.
    
    Attributes:
        camera_manager: Handles camera operations
        model_manager: Manages the SAM model
        frame_processor: Processes frames with effects
        web_server: Handles HTTP requests
        effects_manager: Manages visual effects
        shutdown_event: Controls application shutdown
        metrics: Global application metrics
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApplicationState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize application state if not already initialized"""
        if self._initialized:
            return
            
        self.camera_manager = None
        self.model_manager = None
        self.frame_processor = None
        self.web_server = None
        self.effects_manager = None  # Added missing effects_manager initialization
        self.shutdown_event = threading.Event()
        self.metrics = ProcessingMetrics()
        
        # Component status tracking
        self._component_status = {
            'camera': False,
            'model': False,
            'processor': False,
            'web': False,
            'effects': False  # Added effects status tracking
        }
        
        # Thread-safe status lock
        self._status_lock = threading.Lock()
        self._initialized = True
    
    def set_component_status(self, component: str, status: bool) -> None:
        """
        Update component initialization status
        
        Args:
            component: Name of the component
            status: Whether component is initialized successfully
        """
        with self._status_lock:
            if component in self._component_status:
                self._component_status[component] = status
    
    def get_component_status(self, component: str) -> bool:
        """
        Get component initialization status
        
        Args:
            component: Name of the component
            
        Returns:
            bool: Component initialization status
        """
        with self._status_lock:
            return self._component_status.get(component, False)
    
    def all_components_ready(self) -> bool:
        """Check if all components are initialized successfully"""
        with self._status_lock:
            return all(self._component_status.values())
    
    def reset(self) -> None:
        """Reset application state for clean restart"""
        with self._status_lock:
            self.camera_manager = None
            self.model_manager = None
            self.frame_processor = None
            self.web_server = None
            self.effects_manager = None  # Added effects_manager reset
            self.shutdown_event.clear()
            self.metrics = ProcessingMetrics()
            self._component_status = {k: False for k in self._component_status}

# Global application state instance
app_state = ApplicationState()