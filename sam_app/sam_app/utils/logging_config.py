import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

class LoggingConfig:
    """
    Configures application-wide logging with file and console outputs.
    Provides structured logging with different levels and formats for
    different handlers.
    """
    
    def __init__(self):
        self.log_dir = "logs"
        self.max_bytes = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

    def configure(self, app_name: str = "sam_app") -> None:
        """
        Configure logging system with console and file handlers.
        
        Args:
            app_name: Name of the application for log files
        """
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)

        # Add file handlers
        general_file_handler = self._create_file_handler(
            f"{app_name}.log",
            self.file_level
        )
        root_logger.addHandler(general_file_handler)

        error_file_handler = self._create_file_handler(
            f"{app_name}_error.log",
            logging.ERROR
        )
        root_logger.addHandler(error_file_handler)

    def _create_console_handler(self) -> logging.Handler:
        """
        Create console handler with colored output.
        
        Returns:
            logging.Handler: Configured console handler
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self._create_colored_formatter())
        return console_handler

    def _create_file_handler(
        self, 
        filename: str, 
        level: int
    ) -> logging.Handler:
        """
        Create rotating file handler.
        
        Args:
            filename: Name of log file
            level: Logging level for handler
            
        Returns:
            logging.Handler: Configured file handler
        """
        file_path = os.path.join(self.log_dir, filename)
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        handler.setLevel(level)
        handler.setFormatter(self._create_file_formatter())
        return handler

    def _create_colored_formatter(self) -> logging.Formatter:
        """
        Create formatter with colored output for console.
        
        Returns:
            logging.Formatter: Colored console formatter
        """
        class ColoredFormatter(logging.Formatter):
            """Logging formatter with color support"""
            
            COLORS = {
                'DEBUG': '\033[0;36m',    # Cyan
                'INFO': '\033[0;32m',     # Green
                'WARNING': '\033[0;33m',  # Yellow
                'ERROR': '\033[0;31m',    # Red
                'CRITICAL': '\033[0;35m', # Magenta
                'RESET': '\033[0m'        # Reset
            }

            def format(self, record: logging.LogRecord) -> str:
                color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                record.color_start = color
                record.color_end = self.COLORS['RESET']
                return super().format(record)

        return ColoredFormatter(
            '%(color_start)s[%(asctime)s] %(levelname)s: %(message)s%(color_end)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _create_file_formatter(self) -> logging.Formatter:
        """
        Create formatter for file output.
        
        Returns:
            logging.Formatter: File formatter
        """
        return logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:

        """
        Get a configured logger instance.

        Args:
            name: Logger name (defaults to root logger if None)

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name) if name else logging.getLogger()

        # Ensure logger has our configured handlers
        if not logger.handlers:
            logger.addHandler(self._create_console_handler())
            logger.addHandler(self._create_file_handler(
                f"{name or 'root'}.log",
                self.file_level
            ))

        return logger

    def cleanup_old_logs(self, days: int = 30) -> None:
        """
        Remove log files older than specified days.

        Args:
            days: Number of days to keep logs
        """
        now = datetime.now()
        for filename in os.listdir(self.log_dir):
            filepath = os.path.join(self.log_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if (now - file_time).days > days:
                    try:
                        os.remove(filepath)
                        logging.info(f"Removed old log file: {filename}")
                    except Exception as e:
                        logging.error(f"Error removing old log file {filename}: {e}")

    def set_log_levels(self, console_level: int, file_level: int) -> None:
        """
        Update logging levels for handlers.

        Args:
            console_level: New console handler level
            file_level: New file handler level
        """
        self.console_level = console_level
        self.file_level = file_level

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
            elif isinstance(handler, logging.FileHandler):
                handler.setLevel(file_level)

# Global logging configuration instance
logging_config = LoggingConfig()
