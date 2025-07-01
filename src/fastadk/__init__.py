"""
FastADK - A developer-friendly framework for building AI agents with Google ADK.

FastADK provides high-level abstractions, declarative APIs, and developer-friendly
tooling for building AI agents. It follows the proven patterns of FastAPI and
FastMCP to dramatically improve developer experience.

Example:
    ```python
    from fastadk import Agent, tool, BaseAgent

    @Agent(model="gemini-2.0", description="Weather assistant")
    class WeatherAgent(BaseAgent):
        @tool
        def get_weather(self, city: str) -> dict:
            '''Fetch current weather for a city.'''
            return {"city": city, "temp": "22°C", "condition": "sunny"}
    ```
"""

__version__ = "0.0.1"
__author__ = "FastADK Team"
__email__ = "team@fastadk.dev"
__license__ = "MIT"

# Core imports - will be implemented in later phases
from .core.exceptions import FastADKError

# Version information
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "FastADKError",
]
