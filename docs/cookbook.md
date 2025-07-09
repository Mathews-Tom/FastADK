# FastADK Cookbook

This cookbook provides practical recipes for common tasks when working with FastADK. Each recipe includes step-by-step instructions and code snippets that you can adapt for your own projects.

## Table of Contents

- [Basic Agent Setup](#basic-agent-setup)
- [Adding Custom Tools](#adding-custom-tools)
- [Working with Context and Memory](#working-with-context-and-memory)
- [Implementing Retry Logic](#implementing-retry-logic)
- [Token and Cost Management](#token-and-cost-management)
- [Customizing Providers](#customizing-providers)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Deploying as an API](#deploying-as-an-api)
- [Observability and Monitoring](#observability-and-monitoring)
- [Testing Strategies](#testing-strategies)

## Basic Agent Setup

### Recipe: Creating a Simple Agent

```python
from fastadk.core.agent import BaseAgent

class SimpleAgent(BaseAgent):
    """A simple FastADK agent."""
    
    _description = "A basic demo agent"
    _model_name = "gpt-3.5-turbo"  # Or other models like "gpt-4", "claude-2", etc.
    _provider = "openai"  # Or "anthropic", "google", etc.
    
    async def run(self, prompt: str) -> str:
        """Run the agent with the given prompt."""
        return await super().run(prompt)
```

### Recipe: Running Your Agent

```python
import asyncio
from your_module import SimpleAgent

async def main():
    agent = SimpleAgent()
    response = await agent.run("Tell me about artificial intelligence.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Recipe: Using the CLI

```bash
# Run your agent in interactive mode
uv run -m fastadk run path/to/your_agent.py

# Start a REPL session
uv run -m fastadk repl --module path/to/your_agent.py

# Serve your agent as an API
uv run -m fastadk serve path/to/your_agent.py
```

## Adding Custom Tools

### Recipe: Creating a Custom Tool

```python
from fastadk.core.agent import BaseAgent, tool

class WeatherAgent(BaseAgent):
    """An agent that can check the weather."""
    
    _description = "Weather information assistant"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    @tool
    async def get_weather(self, location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location.
        
        Args:
            location: City, state, or country
            unit: Temperature unit (celsius or fahrenheit)
            
        Returns:
            Current weather information
        """
        # In a real implementation, you would call a weather API
        return f"The weather in {location} is sunny and 25°{unit[0].upper()}"
```

### Recipe: Tool with Error Handling

```python
import aiohttp
from fastadk.core.agent import BaseAgent, tool
from fastadk.core.exceptions import ToolError

class StockAgent(BaseAgent):
    """An agent that can check stock prices."""
    
    _description = "Stock information assistant"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    @tool
    async def get_stock_price(self, symbol: str) -> str:
        """Get the current stock price for a company.
        
        Args:
            symbol: Stock ticker symbol (e.g., AAPL)
            
        Returns:
            Current stock price information
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.example.com/stocks/{symbol}") as response:
                    if response.status != 200:
                        raise ToolError(f"Failed to get stock data: {response.status}")
                    data = await response.json()
                    return f"The current price of {symbol} is ${data['price']:.2f}"
        except aiohttp.ClientError as e:
            raise ToolError(f"Network error when fetching stock data: {str(e)}")
```

## Working with Context and Memory

### Recipe: Custom Context Policy

```python
from fastadk.core.agent import BaseAgent
from fastadk.core.context_policy import ContextPolicy
from fastadk.core.context import Context

class MostRecentPolicy(ContextPolicy):
    """Keep only the most recent N messages."""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        
    async def apply(self, context: Context) -> Context:
        """Apply the policy to the context."""
        if len(context.messages) <= self.max_messages:
            return context
            
        # Keep only the most recent messages
        context.messages = context.messages[-self.max_messages:]
        return context

class MyAgent(BaseAgent):
    """Agent with custom context policy."""
    
    _description = "Agent with custom memory management"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    def __init__(self):
        super().__init__()
        self.context_policies = [MostRecentPolicy(max_messages=5)]
```

### Recipe: Using Vector Memory

```python
from fastadk.core.agent import BaseAgent
from fastadk.memory.vector import VectorMemory

class KnowledgeAgent(BaseAgent):
    """Agent with vector memory for better recall."""
    
    _description = "Knowledge base agent with semantic search"
    _model_name = "gpt-4"
    _provider = "openai"
    
    def __init__(self):
        super().__init__()
        # Initialize with vector memory
        self.memory_backend = VectorMemory(
            collection_name="knowledge_base",
            embedding_model="text-embedding-ada-002"
        )
        
    async def initialize(self):
        """Add initial knowledge to the agent's memory."""
        await self.memory_backend.add("Python is a high-level programming language.")
        await self.memory_backend.add("FastADK is a framework for building AI agents.")
```

### Recipe: Using Redis Memory Backend

```python
from fastadk.core.agent import BaseAgent
from fastadk.memory.redis import RedisMemory

class PersistentAgent(BaseAgent):
    """Agent with persistent memory using Redis."""
    
    _description = "Agent with persistent memory"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    def __init__(self):
        super().__init__()
        # Initialize with Redis memory
        self.memory_backend = RedisMemory(
            redis_url="redis://localhost:6379/0",
            ttl_seconds=3600 * 24  # 24 hours
        )
```

## Implementing Retry Logic

### Recipe: Automatic Retries for Network Issues

```python
from fastadk.core.agent import BaseAgent
from fastadk.core.retry import RetryStrategy, exponential_backoff

class ReliableAgent(BaseAgent):
    """Agent with robust retry logic."""
    
    _description = "Agent with automatic retries"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    def __init__(self):
        super().__init__()
        # Configure retry strategy
        self.retry_strategy = RetryStrategy(
            max_retries=3,
            backoff_func=exponential_backoff,
            retry_on=[
                "connection_error",
                "timeout_error",
                "server_error"
            ]
        )
```

## Token and Cost Management

### Recipe: Setting Token Budgets

```python
from fastadk.core.agent import BaseAgent
from fastadk.tokens.models import TokenBudget

class BudgetedAgent(BaseAgent):
    """Agent with token budget controls."""
    
    _description = "Cost-controlled agent"
    _model_name = "gpt-4"
    _provider = "openai"
    
    def __init__(self):
        super().__init__()
        # Set a budget per conversation and per request
        self.token_budget = TokenBudget(
            max_tokens_per_session=10000,  # 10k tokens per conversation
            max_tokens_per_request=2000,   # 2k tokens per request
            action_on_exceed="warn"        # or "error" to raise an exception
        )
```

### Recipe: Tracking Token Usage

```python
from fastadk.core.agent import BaseAgent
from fastadk.observability.metrics import report_metric

class TrackedAgent(BaseAgent):
    """Agent that tracks token usage."""
    
    _description = "Agent with token tracking"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    async def run(self, prompt: str) -> str:
        """Run the agent and report token usage."""
        response = await super().run(prompt)
        
        # Report token usage as metrics
        if hasattr(self, "_token_usage"):
            report_metric("prompt_tokens", self._token_usage.prompt_tokens)
            report_metric("completion_tokens", self._token_usage.completion_tokens)
            report_metric("total_tokens", self._token_usage.total_tokens)
            
            # Estimate cost
            cost = self._token_usage.estimate_cost()
            report_metric("estimated_cost", cost)
            
        return response
```

## Customizing Providers

### Recipe: Using Multiple Providers

```python
from fastadk.core.agent import BaseAgent
from fastadk.providers.litellm import LiteLLMProvider

class MultiProviderAgent(BaseAgent):
    """Agent that can use different LLM providers."""
    
    _description = "Multi-provider agent"
    
    def __init__(self, provider_name: str, model_name: str):
        super().__init__()
        self._provider = provider_name
        self._model_name = model_name
        
        # Configure LiteLLM provider for flexibility
        self.model = LiteLLMProvider(
            model_name=self._model_name,
            provider=self._provider
        )

# Usage examples:
# agent1 = MultiProviderAgent("openai", "gpt-4")
# agent2 = MultiProviderAgent("anthropic", "claude-2")
# agent3 = MultiProviderAgent("google", "gemini-pro")
```

### Recipe: Creating a Custom Provider

```python
from typing import Dict, Any
from fastadk.core.agent import BaseAgent
from fastadk.providers.base import ModelProvider

class MyCustomProvider(ModelProvider):
    """Custom LLM provider implementation."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        # Initialize your custom client here
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        # Implement your custom API call here
        # ...
        return "Response from custom provider"
        
    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM."""
        # Implement streaming logic
        # ...
        yield "Streaming response from custom provider"
        
    async def health_check(self) -> Dict[str, Any]:
        """Check if the provider is healthy."""
        return {"status": "ok", "latency_ms": 150}

class CustomAgent(BaseAgent):
    """Agent using a custom provider."""
    
    _description = "Agent with custom LLM provider"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.model = MyCustomProvider(api_key=api_key)
```

## Multi-Agent Orchestration

### Recipe: Creating a Workflow with Multiple Agents

```python
from fastadk.core.agent import BaseAgent
from fastadk.core.workflow import Workflow

class ResearchAgent(BaseAgent):
    """Agent for research tasks."""
    _description = "Research assistant"
    _model_name = "gpt-4"
    _provider = "openai"

class WritingAgent(BaseAgent):
    """Agent for writing tasks."""
    _description = "Writing assistant"
    _model_name = "gpt-4"
    _provider = "openai"

class ProjectManager:
    """Orchestrates multiple agents in a workflow."""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.writing_agent = WritingAgent()
        self.workflow = Workflow()
        
    async def create_report(self, topic: str) -> str:
        """Create a complete report on a topic."""
        # Define workflow steps
        research_step = self.workflow.add_step(
            self.research_agent.run,
            args=[f"Research key facts about {topic}"]
        )
        
        outline_step = self.workflow.add_step(
            self.writing_agent.run,
            args=[f"Create an outline for a report on {topic} using this research: {research_step.result}"],
            depends_on=[research_step]
        )
        
        writing_step = self.workflow.add_step(
            self.writing_agent.run,
            args=[f"Write a full report following this outline: {outline_step.result}"],
            depends_on=[outline_step]
        )
        
        # Execute the workflow
        results = await self.workflow.execute()
        return results[writing_step.id]
```

## Deploying as an API

### Recipe: Creating a FastAPI Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastadk.core.agent import BaseAgent

# Define your agent
class MyAgent(BaseAgent):
    """Example agent for API deployment."""
    _description = "API-deployed agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"

# Initialize FastAPI and the agent
app = FastAPI(title="FastADK Agent API")
agent = MyAgent()

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str
    session_id: str = None

class QueryResponse(BaseModel):
    response: str
    usage: Dict[str, Any] = None

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Process a query with the agent."""
    try:
        response = await agent.run(request.prompt)
        
        # Include token usage if available
        usage = None
        if hasattr(agent, "_token_usage"):
            usage = {
                "prompt_tokens": agent._token_usage.prompt_tokens,
                "completion_tokens": agent._token_usage.completion_tokens,
                "total_tokens": agent._token_usage.total_tokens
            }
        
        return QueryResponse(response=response, usage=usage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Recipe: Using FastADK's Built-in API Server

```python
# agent.py
from fastadk.core.agent import BaseAgent

class MyApiAgent(BaseAgent):
    """Agent designed for API deployment."""
    _description = "API service agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"

# Command to run the API server:
# uv run -m fastadk serve agent.py
```

## Observability and Monitoring

### Recipe: Setting Up Structured Logging

```python
from fastadk.core.agent import BaseAgent
from fastadk.observability.logger import setup_logging

# Configure logging
setup_logging(
    level="INFO",
    format="json",
    log_to_file=True,
    log_file_path="./logs/agent.log"
)

class MonitoredAgent(BaseAgent):
    """Agent with enhanced logging."""
    _description = "Observable agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
```

### Recipe: Adding OpenTelemetry Tracing

```python
from fastadk.core.agent import BaseAgent
from fastadk.observability.tracing import setup_tracing

# Configure tracing
setup_tracing(
    service_name="my-fastadk-agent",
    exporter="otlp",  # or "console", "jaeger", etc.
    endpoint="http://collector:4317"
)

class TracedAgent(BaseAgent):
    """Agent with OpenTelemetry tracing."""
    _description = "Traced agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
```

### Recipe: Exporting Prometheus Metrics

```python
from fastadk.core.agent import BaseAgent
from fastadk.observability.metrics import setup_metrics, report_metric

# Configure metrics
setup_metrics(
    service_name="my-fastadk-agent",
    export_to_prometheus=True,
    prometheus_port=8000
)

class MetricsAgent(BaseAgent):
    """Agent that exports metrics."""
    _description = "Metrics-enabled agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    async def run(self, prompt: str) -> str:
        """Run the agent and record metrics."""
        start_time = time.time()
        response = await super().run(prompt)
        
        # Record custom metrics
        execution_time = time.time() - start_time
        report_metric("agent_execution_time_seconds", execution_time)
        report_metric("agent_requests_total", 1)
        
        return response
```

## Testing Strategies

### Recipe: Unit Testing with Mock LLM

```python
import pytest
from fastadk.core.agent import BaseAgent
from fastadk.testing.utils import MockLLMProvider

class SimpleAgent(BaseAgent):
    """Simple agent for testing."""
    _description = "Test agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"

@pytest.fixture
def mock_agent():
    """Create an agent with a mock LLM provider."""
    agent = SimpleAgent()
    # Replace the real provider with a mock
    agent.model = MockLLMProvider(
        responses=["This is a mocked response."]
    )
    return agent

@pytest.mark.asyncio
async def test_agent_run(mock_agent):
    """Test that the agent returns the expected response."""
    response = await mock_agent.run("Hello, agent!")
    assert response == "This is a mocked response."
    assert mock_agent.model.call_count == 1
```

### Recipe: Integration Testing with Tool Mocks

```python
import pytest
from fastadk.core.agent import BaseAgent, tool
from fastadk.testing.utils import MockLLMProvider, mock_tool

class WeatherAgent(BaseAgent):
    """Agent with a weather tool."""
    _description = "Weather agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    @tool
    async def get_weather(self, location: str) -> str:
        """Get weather for a location."""
        # In production, this would call a real API
        return f"Sunny and 75°F in {location}"

@pytest.fixture
def mock_weather_agent():
    """Create an agent with mocked components."""
    agent = WeatherAgent()
    # Mock the LLM provider
    agent.model = MockLLMProvider(
        responses=["The weather in Seattle is sunny."]
    )
    # Mock the weather tool
    mock_tool(agent, "get_weather", lambda location: f"Mocked weather for {location}")
    return agent

@pytest.mark.asyncio
async def test_weather_agent(mock_weather_agent):
    """Test the weather agent with mocked components."""
    response = await mock_weather_agent.run("What's the weather in Seattle?")
    assert "sunny" in response.lower()
    # Verify the tool was called
    assert mock_weather_agent.tools_used == ["get_weather"]
```

---

These recipes should help you get started with common FastADK patterns and practices. For more detailed information on specific components, refer to the relevant sections in the API documentation.

---

Do you have a useful pattern or recipe that should be included here? Share it with us on [GitHub Discussions](https://github.com/Mathews-Tom/FastADK/discussions) or consider contributing to our documentation!
