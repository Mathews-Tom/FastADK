"""
OpenTelemetry tracing for FastADK.

This module provides OpenTelemetry tracing capabilities for LLM calls, tool calls,
context loading, and memory operations.
"""

import functools
import os
from typing import Any, Callable, Optional, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.span import Span

from fastadk.observability.logger import logger

# Try to import optional OpenTelemetry SDK components
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _HAS_OTLP = True
except ImportError:
    _HAS_OTLP = False

# Type variables for decorator typing
F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R")


class TracingManager:
    """Manages OpenTelemetry tracing for FastADK."""

    def __init__(self) -> None:
        """Initialize the tracing manager."""
        self._initialized = False
        self._tracer = None

    def configure(
        self,
        service_name: str = "fastadk",
        enable_console_exporter: bool = False,
        enable_otlp_exporter: bool = False,
        otlp_endpoint: Optional[str] = None,
    ) -> None:
        """Configure the OpenTelemetry tracer.

        Args:
            service_name: The name of the service
            enable_console_exporter: Whether to enable the console exporter
            enable_otlp_exporter: Whether to enable the OTLP exporter
            otlp_endpoint: The endpoint for the OTLP exporter
        """
        if self._initialized:
            logger.warning("Tracing is already initialized")
            return

        # Check if OpenTelemetry SDK is available
        if not _HAS_OTLP:
            logger.warning(
                "OpenTelemetry SDK not available. Install with: uv add opentelemetry-sdk opentelemetry-exporter-otlp"
            )
            self._tracer = trace.get_tracer(__name__, service_name)
            self._initialized = True
            return

        # Create a resource with service info
        resource = Resource.create({"service.name": service_name})

        # Create a tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Add console exporter if enabled
        if enable_console_exporter:
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Add OTLP exporter if enabled
        if enable_otlp_exporter:
            endpoint = otlp_endpoint or os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set the tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Get a tracer
        self._tracer = trace.get_tracer(__name__, service_name)
        self._initialized = True
        logger.info("Tracing initialized", service_name=service_name)

    @property
    def tracer(self) -> Any:
        """Get the tracer.

        Returns:
            The tracer
        """
        if not self._initialized:
            self.configure()
        return self._tracer

    def create_span(self, name: str, attributes: Optional[dict] = None) -> Span:
        """Create a new span.

        Args:
            name: The name of the span
            attributes: Attributes to add to the span

        Returns:
            The created span
        """
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span

    def trace(
        self, span_name: Optional[str] = None, attributes: Optional[dict] = None
    ) -> Callable[[F], F]:
        """Decorator to trace a function.

        Args:
            span_name: The name of the span
            attributes: Attributes to add to the span

        Returns:
            The decorated function
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Use the function name if span_name is not provided
                name = span_name or f"{func.__module__}.{func.__qualname__}"

                # Create span attributes
                span_attributes = attributes.copy() if attributes else {}

                # Start a new span
                with self.tracer.start_as_current_span(name) as span:
                    # Add attributes to the span
                    for key, value in span_attributes.items():
                        span.set_attribute(key, value)

                    try:
                        # Call the function
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except (ValueError, TypeError, KeyError, AttributeError) as e:
                        # Handle common exceptions
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    except Exception as e:  # pylint: disable=broad-except
                        # Handle other exceptions
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return cast(F, wrapper)

        return decorator


# Create a global instance of the tracing manager
tracer = TracingManager()
