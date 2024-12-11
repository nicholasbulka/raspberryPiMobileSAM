from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
import logging
import psutil
import os
from typing import Optional, Dict, Any
from ..core.app_state import app_state
from ..effects.effects_parameters import EffectParameters

logger = logging.getLogger(__name__)

class WebServer:
    """
    Handles web server operations and client communication. Implements routes
    for streaming video, controlling effects, and monitoring system status.
    
    This class is responsible for:
    - Setting up Flask routes
    - Handling client requests
    - Managing effect controls
    - Providing system monitoring endpoints
    """
    
    def __init__(self, app: Flask, app_state):
        """
        Initialize web server with Flask app and application state.
        
        Args:
            app: Flask application instance
            app_state: Application state manager
        """
        self.app = app
        self.app_state = app_state
        self.effects_manager = app_state.effects_manager
        self.setup_routes()
        
        # Configure response headers
        self.default_headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        # Register with application state
        app_state.web_server = self

    def setup_routes(self) -> None:
        """Configure Flask routes with error handling"""
        
        @self.app.route('/')
        def index():
            """Serve main application page"""
            return render_template(
                'index.html',
                effects=self.effects_manager.get_effects_list()
            )

        @self.app.route('/stream')
        def stream():
            """Handle video stream requests"""
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
            """
            Set current visual effect

            Args:
                effect_name: Name of effect to apply
            """
            try:
                self.effects_manager.set_effect(effect_name)
                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error setting effect: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/effects_params', methods=['POST'])
        def update_effect_params():
            """Update effect parameters with validation"""
            try:
                params = EffectParameters(request.json)
                self.effects_manager.update_params(params)
                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
                return jsonify({"error": str(e)}), 500

        # In web_server.py
        @self.app.route('/debug_info')
        def get_debug_info():
            """Get comprehensive system debug information"""
            try:
                # Get current frame masks
                current_masks = []
                current_scores = []
                with self.app_state.camera_manager.frame_lock:
                    if self.app_state.camera_manager.current_frame is not None:
                        current_masks, current_scores = self.app_state.model_manager.predict(
                            self.app_state.camera_manager.current_frame
                        )

                debug_info = {
                    'effects': self.effects_manager.get_debug_info(),
                    'camera': self.app_state.camera_manager.get_metrics(),
                    'model': {
                        **self.app_state.model_manager.get_metrics(),
                        'current_masks_count': len(current_masks) if current_masks is not None else 0,
                        'current_scores': current_scores.tolist() if current_scores is not None else []
                    },
                    'processor': self.app_state.frame_processor.get_metrics(),
                    'system': self.get_system_metrics()
                }
                logger.debug(f"Debug info: {debug_info}")
                return jsonify(debug_info)
            except Exception as e:
                logger.error(f"Error getting debug info: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/health')
        def health_check():
            """System health check endpoint"""
            try:
                health_status = {
                    'status': 'healthy' if self.check_system_health() else 'degraded',
                    'components': {
                        'camera': self.app_state.get_component_status('camera'),
                        'model': self.app_state.get_component_status('model'),
                        'processor': self.app_state.get_component_status('processor'),
                        'web': True
                    }
                }
                return jsonify(health_status)
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                return jsonify({'status': 'error', 'error': str(e)}), 500

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
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.

        Returns:
            dict: System metrics including CPU, memory, and disk usage
        """
        try:
            # Get CPU metrics
            cpu_metrics = {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }

            # Get memory metrics
            memory = psutil.virtual_memory()
            memory_metrics = {
                'total': memory.total / (1024 * 1024 * 1024),  # GB
                'available': memory.available / (1024 * 1024 * 1024),  # GB
                'percent': memory.percent
            }

            # Get disk metrics
            disk = psutil.disk_usage('/')
            disk_metrics = {
                'total': disk.total / (1024 * 1024 * 1024),  # GB
                'free': disk.free / (1024 * 1024 * 1024),  # GB
                'percent': disk.percent
            }

            # Get process metrics
            process = psutil.Process(os.getpid())
            process_metrics = {
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads()
            }

            return {
                'cpu': cpu_metrics,
                'memory': memory_metrics,
                'disk': disk_metrics,
                'process': process_metrics
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

    def check_system_health(self) -> bool:
        """
        Check overall system health.

        Returns:
            bool: True if system is healthy, False otherwise
        """
        try:
            metrics = self.get_system_metrics()

            # Define health thresholds
            thresholds = {
                'cpu_percent': 90,
                'memory_percent': 90,
                'disk_percent': 90
            }

            # Check CPU usage
            if metrics.get('cpu', {}).get('percent', 0) > thresholds['cpu_percent']:
                logger.warning("CPU usage above threshold")
                return False

            # Check memory usage
            if metrics.get('memory', {}).get('percent', 0) > thresholds['memory_percent']:
                logger.warning("Memory usage above threshold")
                return False

            # Check disk usage
            if metrics.get('disk', {}).get('percent', 0) > thresholds['disk_percent']:
                logger.warning("Disk usage above threshold")
                return False

            # Check component status
            if not self.app_state.all_components_ready():
                logger.warning("Not all components are ready")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return False

