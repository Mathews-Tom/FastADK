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
# Install UV (recommended package manager for Python)
pip install uv

# Install FastADK with UV
uv pip install fastadk
```

FastADK is now available on PyPI, making installation simple and straightforward! We recommend using [UV](https://github.com/astral-sh/uv) for significantly faster and more reliable package management.

[See full installation instructions →](getting-started/installation.md)

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

# Run with: uv run app.py
# Or serve HTTP API: uv run -m uvicorn app:app --reload
```

## Next Steps

- [Installation](getting-started/installation.md): Detailed installation instructions
- [Quick Start](getting-started/quick-start.md): Create your first agent in minutes
- [System Overview](system-overview.md): Comprehensive explanation of FastADK's architecture and benefits
- [Concepts](concepts/agents.md): Learn about key FastADK concepts
- [Examples](examples/basic/reasoning_demo.md): Real-world examples to learn from

## Join the Community

FastADK is an open-source project, and we welcome contributions of all kinds.

- [GitHub](https://github.com/Mathews-Tom/FastADK): Star us, fork us, contribute!
- [Discord](https://discord.gg/fastadk): Join our community for discussions
- [Twitter](https://twitter.com/fastadk): Follow for updates

## Feedback and Support

We're always looking to improve FastADK based on your feedback! Here are some ways to get help or share your thoughts:

- **[GitHub Discussions](https://github.com/Mathews-Tom/FastADK/discussions)**: Ask questions, share ideas, or showcase what you've built
- **[GitHub Issues](https://github.com/Mathews-Tom/FastADK/issues)**: Report bugs or request features
- **[Community Support](https://discord.gg/fastadk)**: Get help from the community in our Discord server
- **[Email](mailto:team@fastadk.dev)**: Contact the core team directly

Your feedback helps us prioritize features and improvements for future releases!
