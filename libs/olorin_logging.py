"""
Olorin Project Centralized Logging

Provides OlorinLogger class for standardized logging across all components.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class FlushingFileHandler(RotatingFileHandler):
    """File handler that flushes after every log emit (critical for daemon processes)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class OlorinLogger:
    """
    Centralized logger wrapper for Olorin project.

    Provides standardized logging with:
    - Rotating file output (10MB per file, 5 backups)
    - Console output to stdout
    - Automatic flushing for daemon processes
    - Duplicate handler prevention

    Args:
        log_file: Path to log file (required)
        log_level: Logging level (default: 'INFO')
        name: Logger name (default: __name__ of caller)
    """

    def __init__(self, log_file: str, log_level: str = "INFO", name: str = None):
        self.log_file = log_file
        self.log_level = log_level.upper()

        # Configure root logger (industry standard: single source of truth)
        # All loggers (app + libraries) inherit from root
        root_logger = logging.getLogger()

        # Only configure once - if handlers exist, just get our logger and return
        if root_logger.handlers:
            # Already configured, just get our logger
            self._logger = logging.getLogger(name or __name__)
            return

        # Clear any existing handlers to prevent duplicates (defensive programming)
        root_logger.handlers.clear()

        # Ensure log directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create flushing file handler (10MB rotation, 5 backups)
        file_handler = FlushingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Configure root logger - everything inherits from this
        root_logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Get our logger (inherits from root)
        self._logger = logging.getLogger(name or __name__)
        self._logger.info(f"Logging initialized - writing to {log_file}")

    # Expose standard logging methods
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def setLevel(self, level):
        """Change the logging level."""
        self._logger.setLevel(level)

    def getEffectiveLevel(self):
        """Get the effective logging level."""
        return self._logger.getEffectiveLevel()
