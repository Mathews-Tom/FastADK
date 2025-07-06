"""
Custom exception classes for FastADK.

This module provides a comprehensive hierarchy of exception classes that are used
throughout FastADK to report various error conditions with detailed context
and standardized error codes for improved debugging and error handling.
"""

import inspect
import time
import traceback
from typing import Any

import requests


class FastADKError(Exception):
    """
    Base exception class for all FastADK errors.

    All exception classes in FastADK should inherit from this class to
    maintain a consistent exception hierarchy.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize a FastADKError.

        Args:
            message: The error message
            error_code: Optional error code for categorization
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}

        # Format the message with error code if provided
        formatted_message = f"[{error_code}] {message}" if error_code else message
        super().__init__(formatted_message)

    def __repr__(self) -> str:
        """Return a string representation of the error."""
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


class ConfigurationError(FastADKError):
    """
    Raised when there are issues with agent configuration.

    This could be due to missing required settings, invalid configuration values,
    or other configuration-related problems.
    """


class ServiceUnavailableError(FastADKError):
    """
    Raised when an external service is unavailable.

    This could be due to network issues, service outages, or other problems
    preventing communication with external services like APIs or LLM providers.
    """


class AgentError(FastADKError):
    """Raised when there are agent execution issues."""


class ValidationError(FastADKError):
    """Raised when there are validation issues with data."""


class ToolError(FastADKError):
    """Raised when there are tool execution issues."""


class MemoryBackendError(FastADKError):
    """Raised when there are memory backend issues."""


class PluginError(FastADKError):
    """Raised when there are plugin-related issues."""


class AuthenticationError(FastADKError):
    """Raised when there are authentication issues."""


class RateLimitError(FastADKError):
    """Raised when rate limits are exceeded."""


class OperationTimeoutError(FastADKError):
    """Raised when an operation times out."""


class NotFoundError(FastADKError):
    """Raised when a requested resource is not found."""


class OperationError(FastADKError):
    """Raised when an operation fails for any reason."""


class RetryError(FastADKError):
    """Raised when all retry attempts have failed."""


class ServiceConnectionError(FastADKError):
    """Raised when a connection to an external service fails."""


class WorkflowError(FastADKError):
    """Raised when there are errors in workflow execution."""


class OrchestrationError(FastADKError):
    """Raised when there are errors in multi-agent orchestration."""


class ExceptionTracker:
    """
    Tracks exceptions for monitoring and analysis.

    This class provides methods to track, categorize, and analyze exceptions
    that occur during runtime, helping identify patterns and problematic areas.
    """

    # Class-level storage for exception statistics
    _exception_counts: dict[str, int] = {}
    _exception_details: list[dict[str, Any]] = []
    _max_stored_exceptions = 100  # Limit memory usage

    @classmethod
    def track_exception(cls, exception: FastADKError) -> None:
        """
        Track an exception occurrence.

        Args:
            exception: The FastADK exception to track
        """
        # Increment count for this error code
        error_code = exception.error_code or "UNKNOWN"
        cls._exception_counts[error_code] = cls._exception_counts.get(error_code, 0) + 1

        # Store exception details (limiting to prevent memory issues)
        if len(cls._exception_details) >= cls._max_stored_exceptions:
            cls._exception_details.pop(0)  # Remove oldest

        # Add timestamp and store
        cls._exception_details.append(
            {
                "timestamp": time.time(),
                "error_code": error_code,
                "message": exception.message,
                "details": exception.details,
                "exception_type": exception.__class__.__name__,
            }
        )

    @classmethod
    def get_exception_counts(cls) -> dict[str, int]:
        """
        Get counts of exceptions by error code.

        Returns:
            Dictionary mapping error codes to occurrence counts
        """
        return cls._exception_counts.copy()

    @classmethod
    def get_recent_exceptions(cls, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get the most recent exceptions.

        Args:
            limit: Maximum number of exceptions to return

        Returns:
            List of recent exception details
        """
        return cls._exception_details[-limit:]

    @classmethod
    def clear_tracking_data(cls) -> None:
        """Clear all tracked exception data."""
        cls._exception_counts.clear()
        cls._exception_details.clear()

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """
        Get a summary of exception tracking data.

        Returns:
            Summary statistics about tracked exceptions
        """
        total = sum(cls._exception_counts.values())

        # Get top error codes
        top_errors = sorted(
            cls._exception_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Calculate time range if we have exceptions
        time_range = None
        if cls._exception_details:
            oldest = min(e["timestamp"] for e in cls._exception_details)
            newest = max(e["timestamp"] for e in cls._exception_details)
            time_range = newest - oldest

        return {
            "total_exceptions": total,
            "unique_error_codes": len(cls._exception_counts),
            "top_errors": dict(top_errors),
            "tracked_period_seconds": time_range,
        }


class ExceptionTranslator:
    """
    Translates third-party exceptions to FastADK exceptions.

    This class provides methods to convert common external exceptions
    (like those from requests, json, etc.) into appropriate FastADK
    exceptions, ensuring consistent error handling throughout the application.
    """

    # Mapping of exception types to FastADK error types
    _exception_map: dict[type[Exception], type[FastADKError]] = {
        requests.exceptions.Timeout: OperationTimeoutError,
        requests.exceptions.ConnectionError: ServiceUnavailableError,
        requests.exceptions.RequestException: ServiceUnavailableError,
        ValueError: ValidationError,
        TypeError: ValidationError,
        KeyError: NotFoundError,
        FileNotFoundError: NotFoundError,
    }

    @classmethod
    def translate_exception(
        cls,
        exc: Exception,
        default_message: str | None = None,
        default_error_code: str | None = None,
    ) -> FastADKError:
        """
        Convert external exceptions to FastADK exceptions.

        Args:
            exc: The original exception to translate
            default_message: Optional message to use if no specific one is generated
            default_error_code: Optional error code to use if no specific one is generated

        Returns:
            A FastADK exception that wraps the original exception
        """
        exc_type = type(exc)

        # Get the call stack for debugging context
        stack = traceback.extract_stack()[:-1]  # Exclude this function call
        call_info = stack[-1] if stack else None

        # Prepare error details
        details: dict[str, Any] = {
            "exception_type": exc_type.__name__,
            "original_error": str(exc),
        }

        # Add call location if available
        if call_info:
            details["location"] = f"{call_info.filename}:{call_info.lineno}"

        # Find calling function for context
        frame = inspect.currentframe()
        if frame:
            caller_frame = frame.f_back
            if caller_frame:
                caller_info = inspect.getframeinfo(caller_frame)
                details["caller"] = f"{caller_info.function}"

        # Handle requests exceptions specifically
        if isinstance(exc, requests.exceptions.RequestException) and hasattr(
            exc, "response"
        ):
            response = exc.response
            if response:
                details["status_code"] = response.status_code
                try:
                    details["response_body"] = response.json()
                except (
                    ValueError,
                    TypeError,
                ):  # More specific exceptions for JSON parsing
                    details["response_text"] = response.text[:500]  # Limit text size

        # Get appropriate FastADK error type
        error_class = cls._exception_map.get(exc_type, FastADKError)

        # Generate appropriate error code
        if default_error_code:
            error_code = default_error_code
        else:
            # Generate based on exception type
            error_code = f"EXTERNAL_{exc_type.__name__.upper()}"

        # Create message if not provided
        if default_message:
            message = default_message
        else:
            message = f"External error: {str(exc)}"

        # Create the FastADK exception, preserving the original as cause
        fastadk_error = error_class(
            message=message, error_code=error_code, details=details
        )

        # Set the original exception as the cause
        fastadk_error.__cause__ = exc

        return fastadk_error

    @classmethod
    def register_exception_mapping(
        cls, external_exception: type[Exception], fastadk_exception: type[FastADKError]
    ) -> None:
        """
        Register a new exception mapping.

        Args:
            external_exception: The external exception class to translate from
            fastadk_exception: The FastADK exception class to translate to
        """
        cls._exception_map[external_exception] = fastadk_exception
