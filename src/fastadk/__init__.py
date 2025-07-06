"""
FastADK - A developer-friendly framework for building AI agents with Google ADK.

FastADK provides high-level abstractions, declarative APIs, and developer-friendly
tooling for building AI agents. It follows the proven patterns of FastAPI and
FastMCP to dramatically improve developer experience.

Example:
    ```python
    from fastadk import Agent, tool, BaseAgent

    @Agent(model="gemini-1.5-pro", description="Weather assistant")
    class WeatherAgent(BaseAgent):
        @tool
        def get_weather(self, city: str) -> dict:
            '''Fetch current weather for a city.'''
            return {"city": city, "temp": "22°C", "condition": "sunny"}
    ```

    # Serve your agent with FastAPI
    ```python
    from fastadk.api import create_app, registry

    # Register your agents
    registry.register(WeatherAgent)

    # Create FastAPI app
    app = create_app()
    ```
"""

# Import package metadata from __about__.py
from .__about__ import __author__, __email__, __license__, __version__

# Platform adapters
from .adapters import (
    DiscordAgentAdapter,
    SlackAgentAdapter,
    create_discord_agent,
    create_slack_agent,
)

# API imports
from .api.router import create_app, registry

# Core imports
from .core.agent import Agent, BaseAgent, tool
from .core.config import get_settings
from .core.exceptions import (
    AgentError,
    ConfigurationError,
    FastADKError,
    MemoryBackendError,
    OrchestrationError,
    PluginError,
    ToolError,
    ValidationError,
)
from .core.orchestration import OrchestrationResult, OrchestrationStrategy, Orchestrator
from .core.plugin_manager import PluginManager, default_plugin_manager

# Memory backends
from .memory import MemoryBackend, MemoryEntry, get_memory_backend

# Provider interfaces
from .providers import ModelProviderABC

# Training utilities
from .training import (
    DataConverter,
    DataFormat,
    FineTuner,
    FineTuningConfig,
    FineTuningJob,
    FineTuningProvider,
    default_fine_tuner,
)

# Version information
__all__ = [
    # Core classes and decorators
    "Agent",
    "BaseAgent",
    "tool",
    "get_settings",
    # Memory
    "MemoryBackend",
    "MemoryEntry",
    "get_memory_backend",
    # Multi-agent orchestration
    "Orchestrator",
    "OrchestrationResult",
    "OrchestrationStrategy",
    # Platform adapters
    "SlackAgentAdapter",
    "DiscordAgentAdapter",
    "create_slack_agent",
    "create_discord_agent",
    # Plugin system
    "PluginManager",
    "default_plugin_manager",
    # Provider interfaces
    "ModelProviderABC",
    # Training utilities
    "DataConverter",
    "DataFormat",
    "FineTuner",
    "FineTuningConfig",
    "FineTuningJob",
    "FineTuningProvider",
    "default_fine_tuner",
    # API
    "create_app",
    "registry",
    # Exceptions
    "AgentError",
    "ConfigurationError",
    "FastADKError",
    "MemoryBackendError",
    "OrchestrationError",
    "PluginError",
    "ToolError",
    "ValidationError",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
