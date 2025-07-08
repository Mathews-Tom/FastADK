"""
Prometheus metrics for FastADK.

This module provides Prometheus metrics collection and export capabilities.
"""

from typing import Any, Dict, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from fastadk.observability.logger import logger


class MetricsManager:
    """Manages Prometheus metrics for FastADK."""

    def __init__(self) -> None:
        """Initialize the metrics manager."""
        self._metrics: Dict[str, Any] = {}
        self._initialized = False

        # Check if metrics already exist (for tests)
        self._existing_metrics = set(REGISTRY._names_to_collectors.keys())
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Set up default metrics for FastADK."""
        # LLM metrics
        self.counter(
            "fastadk_llm_requests_total",
            "Total number of LLM requests",
            ["model", "provider"],
        )

        self.counter(
            "fastadk_tokens_used_total",
            "Total number of tokens used",
            ["type", "model", "provider"],
        )

        self.counter(
            "fastadk_cost_estimated_total",
            "Total estimated cost of LLM usage",
            ["model", "provider"],
        )

        self.histogram(
            "fastadk_llm_latency_seconds",
            "Latency of LLM requests in seconds",
            ["model", "provider"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        # Tool metrics
        self.counter(
            "fastadk_tool_calls_total", "Total number of tool calls", ["tool_name"]
        )

        self.histogram(
            "fastadk_tool_latency_seconds",
            "Latency of tool calls in seconds",
            ["tool_name"],
            buckets=(0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        )

        # Memory metrics
        self.counter(
            "fastadk_memory_operations_total",
            "Total number of memory operations",
            ["operation", "backend"],
        )

        self.gauge("fastadk_memory_size", "Size of memory in entries", ["backend"])

        # Agent metrics
        self.counter(
            "fastadk_agent_runs_total", "Total number of agent runs", ["agent_id"]
        )

        self.histogram(
            "fastadk_agent_run_duration_seconds",
            "Duration of agent runs in seconds",
            ["agent_id"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )

        self._initialized = True
        logger.info("Default metrics initialized")

    def counter(
        self, name: str, description: str, labels: Optional[list] = None
    ) -> Counter:
        """Create or retrieve a counter metric.

        Args:
            name: The name of the metric
            description: The description of the metric
            labels: The labels for the metric

        Returns:
            The counter metric
        """
        if name not in self._metrics:
            # Check if metric already exists in registry
            if name in self._existing_metrics:
                self._metrics[name] = REGISTRY._names_to_collectors.get(name)
            else:
                try:
                    self._metrics[name] = Counter(name, description, labels or [])
                except ValueError:
                    # If metric already exists, retrieve it
                    self._metrics[name] = REGISTRY._names_to_collectors.get(name)

        return self._metrics[name]

    def gauge(
        self, name: str, description: str, labels: Optional[list] = None
    ) -> Gauge:
        """Create or retrieve a gauge metric.

        Args:
            name: The name of the metric
            description: The description of the metric
            labels: The labels for the metric

        Returns:
            The gauge metric
        """
        if name not in self._metrics:
            # Check if metric already exists in registry
            if name in self._existing_metrics:
                self._metrics[name] = REGISTRY._names_to_collectors.get(name)
            else:
                try:
                    self._metrics[name] = Gauge(name, description, labels or [])
                except ValueError:
                    # If metric already exists, retrieve it
                    self._metrics[name] = REGISTRY._names_to_collectors.get(name)

        return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[list] = None,
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """Create or retrieve a histogram metric.

        Args:
            name: The name of the metric
            description: The description of the metric
            labels: The labels for the metric
            buckets: The buckets for the histogram

        Returns:
            The histogram metric
        """
        if name not in self._metrics:
            # Check if metric already exists in registry
            if name in self._existing_metrics:
                self._metrics[name] = REGISTRY._names_to_collectors.get(name)
            else:
                try:
                    self._metrics[name] = Histogram(
                        name, description, labels or [], buckets=buckets
                    )
                except ValueError:
                    # If metric already exists, retrieve it
                    self._metrics[name] = REGISTRY._names_to_collectors.get(name)

        return self._metrics[name]

    def summary(
        self, name: str, description: str, labels: Optional[list] = None
    ) -> Summary:
        """Create or retrieve a summary metric.

        Args:
            name: The name of the metric
            description: The description of the metric
            labels: The labels for the metric

        Returns:
            The summary metric
        """
        if name not in self._metrics:
            # Check if metric already exists in registry
            if name in self._existing_metrics:
                self._metrics[name] = REGISTRY._names_to_collectors.get(name)
            else:
                try:
                    self._metrics[name] = Summary(name, description, labels or [])
                except ValueError:
                    # If metric already exists, retrieve it
                    self._metrics[name] = REGISTRY._names_to_collectors.get(name)

        return self._metrics[name]

    def get_metric(self, name: str) -> Any:
        """Get a metric by name.

        Args:
            name: The name of the metric

        Returns:
            The metric
        """
        if name not in self._metrics:
            raise ValueError(f"Metric {name} not found")
        return self._metrics[name]

    def generate_latest(self) -> bytes:
        """Generate the latest metrics.

        Returns:
            The latest metrics in Prometheus format
        """
        return generate_latest()

    def content_type(self) -> str:
        """Get the content type for Prometheus metrics.

        Returns:
            The content type
        """
        return CONTENT_TYPE_LATEST


# Create a global instance of the metrics manager
metrics = MetricsManager()
