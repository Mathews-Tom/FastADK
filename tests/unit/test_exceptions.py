"""
Tests for FastADK exception classes.
"""

import pytest

from fastadk.core.exceptions import (
    AgentError,
    ConfigurationError,
    FastADKError,
    MemoryBackendError,
    PluginError,
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
            except ValueError:
                raise FastADKError("Context error")
        except FastADKError as error:
            assert error.message == "Context error"
            assert isinstance(error.__context__, ValueError)
