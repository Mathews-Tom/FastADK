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

__version__ = "0.0.5"  # Updated for Phase 2
__author__ = "FastADK Team"
__email__ = "team@fastadk.dev"
__license__ = "MIT"

# Core imports
# API imports
from .api.router import create_app, registry
from .core.agent import Agent, BaseAgent, ProviderABC, tool
from .core.config import get_settings
from .core.exceptions import (
    AgentError,
    ConfigurationError,
    FastADKError,
    MemoryBackendError,
    PluginError,
    ToolError,
    ValidationError,
)

# Memory backends
from .memory import MemoryBackend, MemoryEntry, get_memory_backend

# Version information
__all__ = [
    # Core classes and decorators
    "Agent",
    "BaseAgent",
    "ProviderABC",
    "tool",
    "get_settings",
    # Memory
    "MemoryBackend",
    "MemoryEntry",
    "get_memory_backend",
    # API
    "create_app",
    "registry",
    # Exceptions
    "AgentError",
    "ConfigurationError",
    "FastADKError",
    "MemoryBackendError",
    "PluginError",
    "ToolError",
    "ValidationError",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
