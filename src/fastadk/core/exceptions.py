"""
Custom exception classes for FastADK.

This module provides a hierarchy of exception classes that are used
throughout FastADK to report various error conditions.
"""


class FastADKError(Exception):
    """
    Base exception class for all FastADK errors.

    All exception classes in FastADK should inherit from this class to
    maintain a consistent exception hierarchy.
    """


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
