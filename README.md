# FastADK

FastADK is an open‑source Python framework that layers ergonomic abstractions over Google ADK and Vertex AI, enabling developers to design, test, and deploy tool‑using LLM agents with minimal boilerplate. Declarative decorators, auto‑generated FastAPI routes, pluggable memory back‑ends, and built‑in observability let teams prototype locally and scale the same code seamlessly to containerised, serverless environments.

## Features

- **Declarative Agent Development**: Create agents with simple `@Agent` and `@tool` decorators
- **Configuration System**: YAML/TOML-based configuration with environment variables support
- **CLI Interface**: Interactive command-line interface for agent development and testing
- **Testing Utilities**: Built-in testing framework for agent validation and simulation
- **Comprehensive Observability**: Logging, metrics, and tracing for monitoring agent behavior

## Quick Start

```python
from fastadk import Agent, BaseAgent, tool

@Agent(model="gemini-1.5-pro", description="Weather assistant")
class WeatherAgent(BaseAgent):
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        return {"city": city, "temp": "22°C", "condition": "sunny"}

# Run the agent
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = WeatherAgent()
        response = await agent.run("What's the weather in London?")
        print(response)
    
    asyncio.run(main())
```

## Configuration

FastADK supports configuration through YAML or TOML files and environment variables:

```yaml
# fastadk.yaml
environment: dev

model:
  provider: gemini
  model_name: gemini-1.5-pro
  api_key_env_var: GEMINI_API_KEY

memory:
  backend_type: inmemory
  ttl_seconds: 3600

telemetry:
  log_level: debug
```

## Testing

FastADK provides a testing framework for agent validation:

```python
from fastadk.testing import AgentTest, test_scenario

class TestWeatherAgent(AgentTest):
    agent = WeatherAgent()
    
    @test_scenario("sunny_weather")
    async def test_sunny_weather(self):
        response = await self.agent.run("What's the weather in London?")
        assert "sunny" in response.lower()
        assert self.agent.tools_used == ["get_weather"]
```

## Installation

```bash
pip install fastadk
```

## License

FastADK is released under the MIT License.