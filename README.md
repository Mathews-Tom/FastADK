# FastADK

FastADK is an open‑source Python framework that makes building LLM-powered agents simple, efficient, and production-ready. It offers declarative APIs, comprehensive observability, and powerful scaling capabilities that enable developers to go from prototype to production with the same codebase.

## Features

### Core Features

- **Declarative Agent Development**: Build agents with `@Agent` and `@tool` decorators
- **Multi-Provider Support**: Easily switch between OpenAI, Anthropic, Google Gemini, and custom providers
- **Token & Cost Tracking**: Built-in visibility into token usage and cost estimation
- **Memory Management**: Sliding window, summarization, and vector store memory backends
- **Async & Parallelism**: True async execution for high performance and concurrency
- **Plugin Architecture**: Extensible system for custom integrations and tools
- **Context Policies**: Advanced context management with customizable strategies
- **Configuration System**: Powerful YAML/environment-based configuration

### Developer Experience

- **CLI Tools**: Interactive REPL, configuration validation, and project scaffolding
- **Debugging & Observability**: Structured logs, metrics, traces, and verbose mode
- **Testing Utilities**: Mock LLMs, simulation tools, and test scenario decorators
- **IDE Support**: VSCode snippets and type hints for better autocompletion
- **Hot Reload**: Development mode with auto-reload for rapid iteration

### Integration & Extension

- **HTTP API**: Auto-generated FastAPI endpoints for all agents
- **Workflow Orchestration**: Build complex multi-agent systems with sequential and parallel execution
- **System Adapters**: Ready-made Discord and Slack integrations
- **Fine-tuning Helpers**: Utilities for model customization and training
- **Batch Processing**: Tooling for high-volume processing

## Quick Start

```python
from fastadk import Agent, BaseAgent, tool

@Agent(model="gemini-1.5-pro", description="Weather assistant")
class WeatherAgent(BaseAgent):
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        # This would typically come from an actual weather API
        return {
            "city": city,
            "current": {
                "temp_c": 22.5,
                "condition": "Partly cloudy",
                "humidity": 65,
                "wind_kph": 15.3
            },
            "forecast": {
                "tomorrow": {"temp_c": 24.0, "condition": "Sunny"},
                "day_after": {"temp_c": 20.0, "condition": "Light rain"}
            }
        }

# Run the agent
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = WeatherAgent()
        response = await agent.run("What's the weather in London?")
        print(response)
    
    asyncio.run(main())
```

## Installation

```bash
pip install fastadk
```

For development, we recommend using [UV](https://github.com/astral-sh/uv) for faster package management:

```bash
# Install uv
pip install uv

# Install FastADK with uv
uv pip install fastadk

# Run examples with uv
uv run -m examples.basic.weather_agent
```

## Workflow Examples

### Context Management with Memory

```python
from fastadk import Agent, BaseAgent
from fastadk.memory import VectorMemoryBackend
from fastadk.core.context_policy import SummarizeOlderPolicy

@Agent(model="gemini-1.5-pro")
class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Set up vector-based memory
        self.memory = VectorMemoryBackend()
        # Summarize older messages when context gets too large
        self.context_policy = SummarizeOlderPolicy(
            threshold_tokens=3000,
            summarizer=self.model
        )
```

### Parallel Tool Execution with Workflow

```python
from fastadk.core.workflow import Workflow, step

@step
async def fetch_weather(city: str):
    # Implementation details...
    return {"city": city, "weather": "sunny"}

@step
async def fetch_news(city: str):
    # Implementation details...
    return {"city": city, "headlines": ["Local event", "Sports update"]}

async def get_city_info(city: str):
    # Run steps in parallel
    results = await Workflow.parallel(
        fetch_weather(city),
        fetch_news(city)
    ).execute()
    
    return {
        "city": city,
        "weather": results[0]["weather"],
        "news": results[1]["headlines"]
    }
```

### Token Usage and Cost Tracking

```python
from fastadk import Agent, BaseAgent
from fastadk.tokens import TokenBudget

@Agent(model="gpt-4")
class BudgetAwareAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Set budget constraints
        self.token_budget = TokenBudget(
            max_tokens_per_session=100000,
            max_cost_per_session=5.0,  # $5.00 USD
            on_exceed="warn"  # Other options: "error", "log"
        )
    
    async def run(self, prompt: str):
        response = await super().run(prompt)
        # Check usage after run
        usage = self.last_run_token_usage
        print(f"Used {usage.total_tokens} tokens (${usage.estimated_cost:.4f})")
        return response
```

### HTTP API with FastAPI

```python
# api.py
from fastapi import FastAPI
from fastadk import Agent, BaseAgent, tool, registry

@Agent(model="gemini-1.5-pro")
class CalculatorAgent(BaseAgent):
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

# Register the agent and create FastAPI app
registry.register(CalculatorAgent)
app = FastAPI()

# Include auto-generated FastADK router
from fastadk.api.router import get_router
app.include_router(get_router(), prefix="/api/agents")

# Run with: uv run -m uvicorn api:app --reload
```

### Discord or Slack Integration

```python
from fastadk import Agent, BaseAgent, tool
from fastadk.adapters.discord import DiscordAdapter

@Agent(model="gemini-1.5-pro")
class HelpfulAssistant(BaseAgent):
    @tool
    def search_knowledge_base(self, query: str) -> str:
        """Search internal knowledge base for information."""
        # Implementation details...
        return "Here's what I found about your question..."

# Connect to Discord
adapter = DiscordAdapter(
    agent=HelpfulAssistant(),
    bot_token="YOUR_DISCORD_BOT_TOKEN",
    channels=["general", "help-desk"],
    prefix="!assist"
)

# Start the bot
if __name__ == "__main__":
    import asyncio
    asyncio.run(adapter.start())
```

## Advanced Features

### Custom Context Policies

```python
from fastadk.core.context_policy import ContextPolicy
from typing import List, Any

class CustomContextPolicy(ContextPolicy):
    """Custom policy that prioritizes questions and important information."""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.important_keywords = ["urgent", "critical", "important"]
    
    async def apply(self, history: List[Any]) -> List[Any]:
        # Implementation that keeps important messages and removes less relevant ones
        # when context size exceeds max_tokens
        # ...
        return filtered_history
```

### Pluggable Provider System

```python
from fastadk.providers.base import ModelProviderABC
from fastadk import registry

class MyCustomProvider(ModelProviderABC):
    """Custom LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # Other initialization...
    
    async def generate(self, prompt: str, **kwargs):
        # Implementation for text generation
        # ...
        return response
    
    async def stream(self, prompt: str, **kwargs):
        # Implementation for streaming response
        # ...
        yield chunk

# Register custom provider
registry.register_provider("my_provider", MyCustomProvider)

# Use custom provider
@Agent(model="my-model", provider="my_provider")
class CustomAgent(BaseAgent):
    # Agent implementation...
```

### Telemetry and Observability

```python
from fastadk.observability import configure_logging, configure_metrics

# Configure structured JSON logging
configure_logging(
    level="INFO",
    format="json",
    redact_sensitive=True,
    log_file="agent.log"
)

# Configure Prometheus metrics
configure_metrics(
    enable=True,
    port=9090,
    labels={"environment": "production", "service": "agent-api"}
)

# Track custom metrics
from fastadk.observability.metrics import counter, gauge, histogram

# Increment counter when agent is used
counter("agent_calls_total", "Total number of agent calls").inc()

# Record latency of operations
with histogram("agent_latency_seconds", "Latency of agent operations").time():
    # Operation to measure
    result = await agent.run(prompt)
```

## Configuration

FastADK supports configuration through YAML files and environment variables:

```yaml
# fastadk.yaml
environment: production

model:
  provider: gemini
  model_name: gemini-1.5-pro
  api_key_env_var: GEMINI_API_KEY
  timeout_seconds: 30
  retry_attempts: 3

memory:
  backend_type: redis
  connection_string: ${REDIS_URL}
  ttl_seconds: 3600
  namespace: "my-agent"

context:
  policy: "summarize_older"
  max_tokens: 8000
  window_size: 10

observability:
  log_level: info
  metrics_enabled: true
  tracing_enabled: true
  redact_patterns:
    - "api_key=([a-zA-Z0-9-_]+)"
    - "password=([^&]+)"
```

## Testing

FastADK provides comprehensive testing tools:

```python
from fastadk.testing import AgentTest, test_scenario, MockModel

class TestWeatherAgent(AgentTest):
    agent = WeatherAgent()
    
    def setup_method(self):
        # Replace real model with mock for testing
        self.agent.model = MockModel(responses=[
            "The weather in London is currently sunny with a temperature of 22°C."
        ])
    
    @test_scenario("basic_weather_query")
    async def test_basic_weather_query(self):
        response = await self.agent.run("What's the weather in London?")
        
        # Assertions
        assert "sunny" in response.lower()
        assert "22°c" in response.lower()
        assert self.agent.tools_used == ["get_weather"]
        assert self.agent.total_tokens < 1000
```

## CLI Commands

FastADK includes a powerful CLI for development and testing:

```bash
# Start interactive REPL with an agent
fastadk repl agent_file.py

# Validate configuration
fastadk config validate

# Initialize a new agent project
fastadk init my-new-agent

# Run an agent with a prompt
fastadk run agent_file.py "What's the weather in London?"

# Start development server with hot reload
fastadk serve agent_api.py --reload
```

## Documentation

- [System Overview](docs/system-overview.md): Detailed architecture and design
- [Getting Started](docs/getting-started/quick-start.md): Build your first agent
- [Examples](examples/): Real-world agent examples
- [API Reference](docs/api/): Detailed API documentation
- [Cookbook](docs/cookbook.md): Common patterns and recipes

## License

FastADK is released under the MIT License.
