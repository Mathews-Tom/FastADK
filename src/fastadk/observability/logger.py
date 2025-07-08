"""
Structured logging for FastADK.

This module provides a structured logger using loguru that outputs JSON formatted logs
for better integration with log analysis tools.
"""

import sys
from typing import Any

from loguru import logger as loguru_logger


class StructuredLogger:
    """A structured logger that outputs JSON formatted logs."""

    def __init__(self) -> None:
        """Initialize the structured logger."""
        # Remove default handler
        loguru_logger.remove()

        # Add JSON handler
        loguru_logger.add(
            sys.stderr,
            serialize=True,  # Output in JSON format
            level="INFO",
            enqueue=True,  # Thread-safe logging
        )

    def configure(self, level: str = "INFO", json_format: bool = True) -> None:
        """Configure the logger with custom settings.

        Args:
            level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_format: Whether to use JSON format for logs
        """
        loguru_logger.remove()

        if json_format:
            loguru_logger.add(
                sys.stderr,
                serialize=True,
                level=level,
                enqueue=True,
            )
        else:
            # Human-readable format for development
            loguru_logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=level,
                enqueue=True,
            )

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit a structured log entry with event name and additional context.

        Args:
            event: The name of the event being logged
            **kwargs: Additional context to include in the log
        """
        loguru_logger.info({"event": event, **kwargs})

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log
        """
        loguru_logger.debug({"message": message, **kwargs})

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log
        """
        loguru_logger.info({"message": message, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log
        """
        loguru_logger.warning({"message": message, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log
        """
        loguru_logger.error({"message": message, **kwargs})

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log
        """
        loguru_logger.critical({"message": message, **kwargs})


# Create a global instance of the structured logger
logger = StructuredLogger()
