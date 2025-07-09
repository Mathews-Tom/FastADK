# Quick Start Guide

This guide will help you create and run your first FastADK agent in just a few minutes.

## Prerequisites

Ensure you've [installed FastADK](installation.md) and have your Google ADK API key set up.

## Create Your First Agent

Let's create a simple weather agent that can tell us the weather for a given city.

Create a new file named `weather_agent.py`:

```python
from fastadk.core import Agent, BaseAgent, tool

@Agent(
    model="gemini-2.0-pro",  # Specifies the LLM to use
    description="Weather assistant that provides forecasts"
)
class WeatherAgent(BaseAgent):
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        # In a real agent, this would call a weather API
        # For this example, we'll return mock data
        return {
            "city": city,
            "temperature": "22°C",
            "condition": "sunny",
            "humidity": "45%"
        }
    
    @tool
    def get_forecast(self, city: str, days: int = 3) -> list:
        """Get weather forecast for multiple days."""
        # Mock forecast data
        return [
            {"day": 1, "condition": "sunny", "temp": "25°C"},
            {"day": 2, "condition": "cloudy", "temp": "22°C"},
            {"day": 3, "condition": "rainy", "temp": "19°C"}
        ][:days]
```

## Run Your Agent in CLI Mode

Run your agent using the FastADK CLI:

```bash
fastadk run weather_agent.py
```

This starts an interactive session where you can chat with your agent. Try asking:

- "What's the weather in Paris?"
- "Can you give me a 5-day forecast for Tokyo?"
- "Should I pack an umbrella for my trip to London?"

The agent will use your defined tools to answer these questions.

## Serve Your Agent as an HTTP API

FastADK can automatically expose your agent as a REST API:

```bash
fastadk serve weather_agent.py
```

This starts a FastAPI server at [http://localhost:8000](http://localhost:8000) with the following endpoints:

- `POST /agents/weather`: Send messages to your agent
- `GET /docs`: Swagger UI for API documentation

You can test the API using curl:

```bash
curl -X POST http://localhost:8000/agents/weather \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in San Francisco?"}'
```

Or using the Swagger UI by opening [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## Adding Memory to Your Agent

Let's enhance our agent to remember previous conversations:

```python
@Agent(
    model="gemini-2.0-pro",
    description="Weather assistant with memory",
    memory_backend="inmemory"  # Enable in-memory storage
)
class WeatherAgentWithMemory(BaseAgent):
    # ... same tools as before ...
    
    @tool
    def remember_preference(self, temperature_unit: str) -> str:
        """Remember the user's preferred temperature unit (C or F)."""
        self.context.set("preferred_unit", temperature_unit)
        return f"I'll remember that you prefer {temperature_unit}"
    
    @tool
    def get_weather_with_preference(self, city: str) -> dict:
        """Get weather using the user's preferred unit if set."""
        preferred_unit = self.context.get("preferred_unit", "C")
        
        # Mock data with unit conversion
        temp = 22 if preferred_unit == "C" else 72
        return {
            "city": city,
            "temperature": f"{temp}°{preferred_unit}",
            "condition": "sunny"
        }
```

Now your agent will remember the user's temperature preference across messages.

## Error Handling

FastADK provides comprehensive error handling. Let's add some error handling to our agent:

```python
from fastadk.core.exceptions import ToolError

@Agent(model="gemini-2.0-pro")
class WeatherAgentWithErrorHandling(BaseAgent):
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        # Validate input
        if not city or len(city) < 2:
            raise ToolError(
                message="City name is too short",
                error_code="INVALID_CITY",
                details={"city": city}
            )
            
        # Implement actual weather lookup
        # For demo, just return mock data
        return {"city": city, "temperature": "22°C"}
```

## Next Steps

Now that you have created your first agent, you can:

- Learn about [agent workflows](../concepts/workflows.md) for complex scenarios
- Add [advanced memory capabilities](../concepts/memory.md) with Redis
- Explore [advanced tool features](../concepts/tools.md) like caching and retries
- Check out the [full examples](../examples/basic/reasoning_demo.md) for more sophisticated implementations
