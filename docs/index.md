# FastADK: The Developer-Friendly Framework for AI Agents

FastADK is an open-source framework that dramatically improves the developer experience when building AI agents with Google's Agent Development Kit (ADK).

## The FastADK Advantage

FastADK follows the proven pattern of FastAPI and other modern frameworks: providing high-level abstractions, declarative APIs, and developer-friendly tooling while leveraging the full power of the underlying platform.

```python
from fastadk.core import Agent, BaseAgent, tool

@Agent(
    model="gemini-2.0-pro", 
    description="Weather assistant that provides forecasts and recommendations"
)
class WeatherAgent(BaseAgent):
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        # Your implementation here
        return {"city": city, "temp": "22°C", "condition": "sunny"}
    
    @tool(cache_ttl=300)  # Cache results for 5 minutes
    def get_forecast(self, city: str, days: int = 5) -> list:
        """Get weather forecast for multiple days."""
        # Your implementation here
        return [
            {"day": 1, "condition": "sunny", "temp": "25°C"},
            {"day": 2, "condition": "cloudy", "temp": "22°C"},
            # More forecast data...
        ]
```

## Key Features

- **Declarative Syntax**: Define agents with `@Agent` and tools with `@tool` decorators
- **Automatic HTTP API**: Serve your agents via FastAPI with zero additional code
- **Memory Management**: Built-in conversation memory with multiple backends
- **Error Handling**: Comprehensive exception framework with meaningful error messages
- **Workflows**: Compose multiple agents to solve complex problems
- **Developer Tools**: CLI for testing, debugging, and deployment

## Designed for Developers

FastADK is built by developers, for developers. We've focused on creating an intuitive, well-documented framework that makes agent development a joy.

- **Minimal Boilerplate**: Accomplish in 10 lines what would take 100+ lines in raw ADK
- **IDE-Friendly**: Complete type hints for excellent editor support
- **Extensive Documentation**: Tutorials, examples, and API references
- **Production Ready**: Built for performance, reliability, and scalability

## Installation

```bash
pip install fastadk
```

## Quick Example

```python
# app.py
from fastadk.core import Agent, BaseAgent, tool

@Agent(model="gemini-2.0-pro")
class MathAgent(BaseAgent):
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

# Run with: fastadk run app.py
# Or serve HTTP API: fastadk serve app.py
```

## Next Steps

- [Installation](getting-started/installation.md): Detailed installation instructions
- [Quick Start](getting-started/quick-start.md): Create your first agent in minutes
- [Tutorial](getting-started/tutorial.md): Step-by-step guide to building a complete agent
- [Examples](examples/weather-agent.md): Real-world examples to learn from

## Join the Community

FastADK is an open-source project, and we welcome contributions of all kinds.

- [GitHub](https://github.com/aetherforge/fastadk): Star us, fork us, contribute!
- [Discord](https://discord.gg/fastadk): Join our community for discussions
- [Twitter](https://twitter.com/fastadk): Follow for updates
