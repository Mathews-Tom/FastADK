"""
Tests for FastADK exception classes.
"""

# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member,protected-access

import pytest
import requests

from fastadk.core.exceptions import (
    AgentError,
    ConfigurationError,
    ExceptionTranslator,
    FastADKError,
    MemoryBackendError,
    NotFoundError,
    OperationTimeoutError,
    PluginError,
    RateLimitError,
    ServiceUnavailableError,
    ToolError,
    ValidationError,
)


class TestFastADKError:
    """Test cases for the base FastADKError class."""

    def test_basic_exception(self):
        """Test basic exception creation and properties."""
        error = FastADKError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}

    def test_exception_with_error_code(self):
        """Test exception with error code."""
        error = FastADKError("Test error", error_code="TEST_001")
        assert str(error) == "[TEST_001] Test error"
        assert error.error_code == "TEST_001"

    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"component": "test", "value": 42}
        error = FastADKError("Test error", details=details)
        assert error.details == details

    def test_exception_repr(self):
        """Test exception string representation."""
        error = FastADKError("Test error", error_code="TEST_001")
        expected = "FastADKError(message='Test error', error_code='TEST_001')"
        assert repr(error) == expected

    def test_exception_inheritance(self):
        """Test that FastADKError inherits from Exception."""
        error = FastADKError("Test error")
        assert isinstance(error, Exception)


class TestSpecificExceptions:
    """Test cases for specific exception classes."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            ConfigurationError,
            AgentError,
            ToolError,
            MemoryBackendError,
            PluginError,
            ValidationError,
        ],
    )
    def test_exception_inheritance(self, exception_class):
        """Test that all specific exceptions inherit from FastADKError."""
        error = exception_class("Test message")
        assert isinstance(error, FastADKError)
        assert isinstance(error, Exception)

    def test_configuration_error(self):
        """Test ConfigurationError specific functionality."""
        error = ConfigurationError("Invalid config", error_code="CONFIG_001")
        assert str(error) == "[CONFIG_001] Invalid config"

    def test_agent_error(self):
        """Test AgentError specific functionality."""
        error = AgentError("Agent failed", details={"agent_id": "test_agent"})
        assert error.details["agent_id"] == "test_agent"

    def test_tool_error(self):
        """Test ToolError specific functionality."""
        error = ToolError("Tool execution failed")
        assert "Tool execution failed" in str(error)

    def test_memory_backend_error(self):
        """Test MemoryBackendError specific functionality."""
        error = MemoryBackendError("Memory backend unavailable")
        assert "Memory backend unavailable" in str(error)

    def test_plugin_error(self):
        """Test PluginError specific functionality."""
        error = PluginError("Plugin loading failed")
        assert "Plugin loading failed" in str(error)

    def test_validation_error(self):
        """Test ValidationError specific functionality."""
        error = ValidationError("Input validation failed")
        assert "Input validation failed" in str(error)


class TestExceptionChaining:
    """Test exception chaining and context."""

    def test_exception_chaining(self):
        """Test that exceptions can be chained properly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise FastADKError("Wrapped error") from e
        except FastADKError as error:
            assert error.message == "Wrapped error"
            assert isinstance(error.__cause__, ValueError)
            assert str(error.__cause__) == "Original error"

    def test_exception_context(self):
        """Test exception context handling."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as exc:
                raise FastADKError("Context error") from exc
        except FastADKError as error:
            assert error.message == "Context error"
            assert isinstance(error.__context__, ValueError)


class TestExceptionTranslator:
    """Test the ExceptionTranslator functionality."""

    def test_translate_value_error(self):
        """Test translation of ValueError to ValidationError."""
        original_error = ValueError("Invalid value")
        translated = ExceptionTranslator.translate_exception(original_error)

        assert isinstance(translated, ValidationError)
        assert translated.error_code == "EXTERNAL_VALUEERROR"
        assert "Invalid value" in translated.message
        assert translated.details["exception_type"] == "ValueError"
        assert translated.__cause__ == original_error

    def test_translate_key_error(self):
        """Test translation of KeyError to NotFoundError."""
        original_error = KeyError("missing_key")
        translated = ExceptionTranslator.translate_exception(original_error)

        assert isinstance(translated, NotFoundError)
        assert "missing_key" in translated.details["original_error"]

    def test_translate_with_custom_message_and_code(self):
        """Test translation with custom message and error code."""
        original_error = TypeError("Type mismatch")
        translated = ExceptionTranslator.translate_exception(
            original_error,
            default_message="Custom error message",
            default_error_code="CUSTOM_ERROR",
        )

        assert translated.message == "Custom error message"
        assert translated.error_code == "CUSTOM_ERROR"
        assert translated.details["exception_type"] == "TypeError"
        assert translated.details["original_error"] == "Type mismatch"

    def test_translate_requests_timeout(self):
        """Test translation of requests.Timeout to OperationTimeoutError."""
        original_error = requests.exceptions.Timeout("Request timed out")
        translated = ExceptionTranslator.translate_exception(original_error)

        assert isinstance(translated, OperationTimeoutError)
        assert "timed out" in translated.message

    def test_translate_requests_connection_error(self):
        """Test translation of requests.ConnectionError to ServiceUnavailableError."""
        original_error = requests.exceptions.ConnectionError("Connection failed")
        translated = ExceptionTranslator.translate_exception(original_error)

        assert isinstance(translated, ServiceUnavailableError)
        assert "Connection failed" in translated.details["original_error"]

    def test_register_custom_mapping(self):
        """Test registering a custom exception mapping."""

        # Create a custom exception type
        class CustomError(Exception):
            pass

        # Register a mapping from CustomError to RateLimitError
        ExceptionTranslator.register_exception_mapping(CustomError, RateLimitError)

        # Test the mapping
        original_error = CustomError("Custom error occurred")
        translated = ExceptionTranslator.translate_exception(original_error)

        assert isinstance(translated, RateLimitError)
        assert "Custom error occurred" in translated.details["original_error"]
