# Agent API Reference

The Agent module is the core of FastADK, providing the main decorators and base classes for creating AI agents.

## Agent Decorator

::: fastadk.core.agent.Agent

## Tool Decorator

::: fastadk.core.agent.tool

## BaseAgent Class

::: fastadk.core.agent.BaseAgent

## Usage Examples

### Basic Agent

```python
from fastadk.core import Agent, BaseAgent, tool

@Agent(model="gemini-2.0-pro")
class SimpleAgent(BaseAgent):
    @tool
    def greet(self, name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"
```

### Advanced Agent Configuration

```python
@Agent(
    model="gemini-2.0-pro", 
    description="Advanced agent with configuration",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    max_retries=3,
    memory_backend="redis",
    memory_ttl=3600,  # 1 hour
    cache_enabled=True
)
class AdvancedAgent(BaseAgent):
    # Agent tools and methods...
    pass
```

### Tool Configuration

```python
@Agent(model="gemini-2.0-pro")
class ToolConfigAgent(BaseAgent):
    @tool
    def simple_tool(self, param: str) -> str:
        """Basic tool without special configuration."""
        return f"Processed: {param}"
    
    @tool(
        cache_ttl=300,  # Cache results for 5 minutes
        timeout=10,     # Timeout after 10 seconds
        retries=3,      # Retry up to 3 times on failure
        retry_delay=1   # Wait 1 second between retries
    )
    def advanced_tool(self, param: str) -> str:
        """Tool with caching, timeout, and retry configuration."""
        return f"Advanced processing: {param}"
```

### Lifecycle Methods

```python
@Agent(model="gemini-2.0-pro")
class LifecycleAgent(BaseAgent):
    def on_initialize(self) -> None:
        """Called when the agent is initialized."""
        self.logger.info("Agent initialized")
        self.setup_counter = 0
    
    def on_message(self, message: str) -> None:
        """Called before processing each message."""
        self.setup_counter += 1
        self.logger.info(f"Processing message. Counter: {self.setup_counter}")
    
    def on_finish(self, result: Any) -> None:
        """Called after processing completes."""
        self.logger.info(f"Processing finished with result: {result}")
```

## API Details

### Agent Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | The LLM model to use, e.g., "gemini-2.0-pro" |
| `description` | `str` | A description of the agent's purpose |
| `max_tokens` | `int` | Maximum tokens in the response |
| `temperature` | `float` | Temperature for sampling (0.0-1.0) |
| `top_p` | `float` | Nucleus sampling parameter (0.0-1.0) |
| `top_k` | `int` | Top-k sampling parameter |
| `max_retries` | `int` | Maximum LLM API call retries |
| `memory_backend` | `str` | Memory backend type ("inmemory", "redis") |
| `memory_ttl` | `int` | Time-to-live for memory entries in seconds |
| `cache_enabled` | `bool` | Enable tool response caching |
| `provider` | `str` | Backend provider ("adk", "langchain") |

### Tool Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Custom name for the tool (defaults to function name) |
| `description` | `str` | Custom description (defaults to docstring) |
| `cache_ttl` | `int` | Time-to-live for cached results in seconds |
| `timeout` | `float` | Maximum execution time in seconds |
| `retries` | `int` | Number of retry attempts on failure |
| `retry_delay` | `float` | Initial delay between retries in seconds |
| `enabled` | `bool` | Whether the tool is enabled |