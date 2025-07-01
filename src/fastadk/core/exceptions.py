"""
FastADK exception classes.

This module defines the exception hierarchy for FastADK, providing structured
error handling across the framework components.
"""

from typing import Any


class FastADKError(Exception):
    """
    Base exception class for all FastADK errors.

    This is the root exception that all other FastADK exceptions inherit from.
    It provides structured error information and supports error codes for
    better error handling in applications.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


class ConfigurationError(FastADKError):
    """Raised when there are configuration-related issues."""

    pass


class AgentError(FastADKError):
    """Raised when there are agent-related issues."""

    pass


class ToolError(FastADKError):
    """Raised when there are tool execution issues."""

    pass


class MemoryError(FastADKError):
    """Raised when there are memory backend issues."""

    pass


class PluginError(FastADKError):
    """Raised when there are plugin-related issues."""

    pass


class ValidationError(FastADKError):
    """Raised when input validation fails."""

    pass


class SecurityError(FastADKError):
    """Raised when security checks fail."""

    pass


class ProviderError(FastADKError):
    """Raised when there are provider backend issues."""

    pass
