"""Tests for observability components."""

from unittest.mock import MagicMock, patch

from fastadk.observability.logger import StructuredLogger
from fastadk.observability.metrics import MetricsManager
from fastadk.observability.redaction import RedactionFilter
from fastadk.observability.tracing import TracingManager


class TestRedactionFilter:
    """Tests for the RedactionFilter class."""

    def test_redact_api_key(self):
        """Test redacting API keys."""
        redaction = RedactionFilter()

        # Test various API key formats
        text = 'api_key="sk-12345abcdef"'
        assert "sk-12345abcdef" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

        text = "apikey: sk_test_51HV9U2J7B3X2K4Y5Z6A7B8C9D0E1F2G3H4I5J6K7L8M9"
        assert "sk_test" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

    def test_redact_credit_card(self):
        """Test redacting credit card numbers."""
        redaction = RedactionFilter()

        text = "My credit card is 4111-1111-1111-1111"
        assert "4111-1111-1111-1111" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

        text = "Card: 4111 1111 1111 1111"
        assert "4111 1111 1111 1111" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

    def test_redact_ssn(self):
        """Test redacting SSNs."""
        redaction = RedactionFilter()

        text = "SSN: 123-45-6789"
        assert "123-45-6789" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

        text = "SSN 123456789"
        assert "123456789" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

    def test_redact_email(self):
        """Test redacting email addresses."""
        redaction = RedactionFilter()

        text = "Contact me at user@example.com"
        assert "user@example.com" not in redaction.redact(text)
        assert "[REDACTED]" in redaction.redact(text)

    def test_redact_dict(self):
        """Test redacting dictionaries."""
        redaction = RedactionFilter()

        data = {
            "api_key": "sk-12345abcdef",
            "user": {"email": "user@example.com", "credit_card": "4111-1111-1111-1111"},
            "message": "This is a test message",
            "tags": ["test", "example"],
        }

        redacted = redaction.redact_dict(data)

        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["user"]["email"] == "[REDACTED]"
        assert redacted["user"]["credit_card"] == "[REDACTED]"
        assert redacted["message"] == "This is a test message"
        assert redacted["tags"] == ["test", "example"]

    def test_custom_patterns(self):
        """Test custom redaction patterns."""
        custom_patterns = [
            r"(project[-_]id):?\s*([A-Za-z0-9-_]+)",
            r"(custom[-_]secret):?\s*([A-Za-z0-9-_]+)",
        ]

        redaction = RedactionFilter(patterns=custom_patterns)

        text = "project_id: abc-123, custom_secret: xyz-456"
        redacted = redaction.redact(text)

        # Just check if the patterns are properly redacted
        assert "[REDACTED]" in redacted
        assert "project_id" in redacted
        assert "custom_secret" in redacted

    def test_disabled_redaction(self):
        """Test disabled redaction."""
        redaction = RedactionFilter(enabled=False)

        text = "api_key='sk-12345abcdef'"
        assert redaction.redact(text) == text

        data = {"api_key": "sk-12345"}
        assert redaction.redact_dict(data) == data


class TestMetricsManager:
    """Tests for the MetricsManager class."""

    def test_counter(self):
        """Test counter metrics."""
        metrics_manager = MetricsManager()

        # Get a counter and increment it
        counter = metrics_manager.counter(
            "test_unique_counter", "Test counter", ["label"]
        )
        counter.labels(label="test").inc()

        # Check that the counter was created
        assert "test_unique_counter" in metrics_manager._metrics

        # Generate metrics and check that they contain the counter
        latest = metrics_manager.generate_latest()
        assert b"test_unique_counter" in latest

    def test_gauge(self):
        """Test gauge metrics."""
        metrics_manager = MetricsManager()

        # Get a gauge and set its value
        gauge = metrics_manager.gauge("test_unique_gauge", "Test gauge", ["label"])
        gauge.labels(label="test").set(42)

        # Check that the gauge was created
        assert "test_unique_gauge" in metrics_manager._metrics

        # Generate metrics and check that they contain the gauge
        latest = metrics_manager.generate_latest()
        assert b"test_unique_gauge" in latest

    def test_default_metrics(self):
        """Test that default metrics are created."""
        metrics_manager = MetricsManager()

        # Check that default metrics were created
        assert "fastadk_llm_requests_total" in metrics_manager._metrics
        assert "fastadk_tokens_used_total" in metrics_manager._metrics
        assert "fastadk_cost_estimated_total" in metrics_manager._metrics
        assert "fastadk_llm_latency_seconds" in metrics_manager._metrics
        assert "fastadk_tool_calls_total" in metrics_manager._metrics
        assert "fastadk_tool_latency_seconds" in metrics_manager._metrics
        assert "fastadk_memory_operations_total" in metrics_manager._metrics
        assert "fastadk_memory_size" in metrics_manager._metrics
        assert "fastadk_agent_runs_total" in metrics_manager._metrics
        assert "fastadk_agent_run_duration_seconds" in metrics_manager._metrics


class TestStructuredLogger:
    """Tests for the StructuredLogger class."""

    @patch("fastadk.observability.logger.loguru_logger")
    def test_emit(self, mock_logger):
        """Test emitting structured logs."""
        logger = StructuredLogger()

        logger.emit("test_event", key1="value1", key2="value2")

        # Check that the logger was called with the correct arguments
        mock_logger.info.assert_called_once()
        log_entry = mock_logger.info.call_args[0][0]
        assert log_entry["event"] == "test_event"
        assert log_entry["key1"] == "value1"
        assert log_entry["key2"] == "value2"

    @patch("fastadk.observability.logger.loguru_logger")
    def test_levels(self, mock_logger):
        """Test different log levels."""
        logger = StructuredLogger()

        logger.debug("Debug message", extra="debug")
        logger.info("Info message", extra="info")
        logger.warning("Warning message", extra="warning")
        logger.error("Error message", extra="error")
        logger.critical("Critical message", extra="critical")

        # Check that the logger was called with the correct arguments for each level
        mock_logger.debug.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()
        mock_logger.critical.assert_called_once()

        debug_entry = mock_logger.debug.call_args[0][0]
        assert debug_entry["message"] == "Debug message"
        assert debug_entry["extra"] == "debug"

        info_entry = mock_logger.info.call_args[0][0]
        assert info_entry["message"] == "Info message"
        assert info_entry["extra"] == "info"


class TestTracingManager:
    """Tests for the TracingManager class."""

    @patch("fastadk.observability.tracing.trace")
    def test_create_span(self, mock_trace):
        """Test creating spans."""
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = TracingManager()
        tracer.configure(service_name="test-service")

        # Create a span
        span = tracer.create_span("test_span", {"key": "value"})

        # Check that the tracer was called with the correct arguments
        mock_tracer.start_span.assert_called_once_with("test_span")

        # Check that span attributes were set
        span.set_attribute.assert_called_once_with("key", "value")

    @patch("fastadk.observability.tracing.trace")
    def test_trace_decorator(self, mock_trace):
        """Test the trace decorator."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = TracingManager()
        tracer.configure(service_name="test-service")

        # Define a function to trace
        @tracer.trace("test_function", {"key": "value"})
        def test_function(arg1, arg2):
            return arg1 + arg2

        # Call the function
        result = test_function(1, 2)

        # Check that the result is correct
        assert result == 3

        # Check that the tracer was called with the correct arguments
        mock_tracer.start_as_current_span.assert_called_once_with("test_function")

        # Check that span attributes were set
        mock_span.set_attribute.assert_called_once_with("key", "value")
