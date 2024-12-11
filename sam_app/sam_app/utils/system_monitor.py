import psutil
import os
import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..core.app_state import app_state

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Holds system performance metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    temperature: float = 0.0
    network_io: Dict[str, int] = None
    process_metrics: Dict[str, float] = None

class SystemMonitor:
    """
    Monitors system resources and performance metrics.
    Provides real-time monitoring of CPU, memory, disk, and network usage.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize system monitor.
        
        Args:
            update_interval: How often to update metrics (seconds)
        """
        self.update_interval = update_interval
        self.metrics = SystemMetrics()
        self.is_running = False
        self.monitor_thread = None
        
        # Initialize network IO baseline
        self.last_network_io = psutil.net_io_counters()
        self.last_update_time = time.time()
        
        # Thresholds for warnings
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 80.0,
            'disk_percent': 80.0,
            'temperature': 80.0
        }

    def start(self) -> None:
        """Start system monitoring in background thread"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop(self) -> None:
        """Stop system monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop updating system metrics"""
        while self.is_running:
            try:
                self._update_metrics()
                self._check_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)

    def _update_metrics(self) -> None:
        """Update all system metrics"""
        try:
            # CPU metrics
            self.metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.disk_percent = disk.percent
            
            # Temperature (Raspberry Pi specific)
            self.metrics.temperature = self._get_cpu_temperature()
            
            # Network IO rates
            current_time = time.time()
            current_net_io = psutil.net_io_counters()
            time_delta = current_time - self.last_update_time
            
            self.metrics.network_io = {
                'bytes_sent': (current_net_io.bytes_sent - self.last_network_io.bytes_sent) / time_delta,
                'bytes_recv': (current_net_io.bytes_recv - self.last_network_io.bytes_recv) / time_delta
            }
            
            self.last_network_io = current_net_io
            self.last_update_time = current_time
            
            # Process metrics
            self.metrics.process_metrics = self._get_process_metrics()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _get_cpu_temperature(self) -> float:
        """
        Get CPU temperature (Raspberry Pi specific).
        
        Returns:
            float: CPU temperature in Celsius
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
            return temp
        except:
            return 0.0

    def _get_process_metrics(self) -> Dict[str, float]:
        """
        Get current process performance metrics.
        
        Returns:
            dict: Process metrics including CPU and memory usage
        """
        try:
            process = psutil.Process(os.getpid())
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds()
            }
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            return {}

    def _check_thresholds(self) -> None:
        """Check metrics against thresholds and log warnings"""
        for metric, threshold in self.thresholds.items():
            current_value = getattr(self.metrics, metric)
            if current_value > threshold:
                logger.warning(
                    f"{metric} above threshold: {current_value:.1f}% (threshold: {threshold}%)"
                )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            dict: Current system metrics
        """
        return {
            'cpu': {
                'percent': self.metrics.cpu_percent,
                'temperature': self.metrics.temperature
            },
            'memory': {
                'percent': self.metrics.memory_percent
            },
            'disk': {
                'percent': self.metrics.disk_percent
            },
            'network': self.metrics.network_io,
            'process': self.metrics.process_metrics
        }

    def set_threshold(self, metric: str, value: float) -> None:
        """
        Update monitoring threshold for a metric.
        
        Args:
            metric: Name of metric
            value: New threshold value
        """
        if metric in self.thresholds:
            self.thresholds[metric] = value
        else:
            logger.warning(f"Unknown metric for threshold: {metric}")

# Global system monitor instance
system_monitor = SystemMonitor()
