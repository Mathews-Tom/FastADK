# FastADK Examples

This directory contains examples demonstrating various features and use cases of the FastADK framework, organized by complexity and focus area.

## Directory Structure

- `basic/`: Simple examples for getting started
- `advanced/`: Complex examples with advanced features
- `ui/`: User interface integrations
- `api/`: HTTP API examples
- `patterns/`: Design patterns and best practices
- `training/`: Fine-tuning and training examples

## Basic Examples

### Weather Agent (`basic/weather_agent.py`)

A simple agent that provides weather information for different locations.

**Features demonstrated:**

- Basic agent setup and configuration
- Simple tool implementation
- Error handling

**Run this example:**

```bash
uv run examples/basic/weather_agent.py
```

### Exception Demo (`basic/exception_demo.py`)

Demonstrates how to handle different types of exceptions in FastADK.

**Features demonstrated:**

- Error handling patterns
- Exception customization
- Fallback strategies

**Run this example:**

```bash
uv run examples/basic/exception_demo.py
```

### Reasoning Demo (`basic/reasoning_demo.py`)

Shows how FastADK can be used for complex reasoning tasks.

**Features demonstrated:**

- Step-by-step reasoning
- Chain-of-thought patterns
- Problem-solving capabilities

**Run this example:**

```bash
uv run examples/basic/reasoning_demo.py
```

### Token Tracking Demo (`basic/token_tracking_demo.py`)

Demonstrates token usage tracking and management.

**Features demonstrated:**

- Token counting
- Cost estimation
- Budget management

**Run this example:**

```bash
uv run examples/basic/token_tracking_demo.py
```

### LiteLLM Demo (`basic/litellm_demo.py`)

Shows integration with LiteLLM for multi-provider support.

**Features demonstrated:**

- Using multiple LLM providers
- Provider fallbacks
- Model configuration

**Run this example:**

```bash
uv run examples/basic/litellm_demo.py
```

## Advanced Examples

### Finance Assistant (`advanced/finance_assistant.py`)

A comprehensive financial assistant agent.

**Features demonstrated:**

- Token budget management
- Complex tool implementations
- Error handling in tools
- JSON response formatting for structured data

**Capabilities:**

- Provide stock price information
- Calculate compound interest
- Calculate mortgage payments
- Estimate taxes
- Calculate retirement savings projections

**Run this example:**

```bash
uv run examples/advanced/finance_assistant.py
```

### Customer Support Agent (`advanced/customer_support.py`)

A customer support agent for a consumer electronics company.

**Features demonstrated:**

- Custom context policy for prioritizing support-related messages
- InMemory backend for session persistence
- Structured knowledge base integration
- Multi-step conversation handling

**Capabilities:**

- Answer product information inquiries
- Check order status
- Process refund requests
- Create support tickets
- Provide technical support from a knowledge base

**Run this example:**

```bash
uv run examples/advanced/customer_support.py
```

### Travel Assistant (`advanced/travel_assistant.py`)

A travel planning assistant.

**Features demonstrated:**

- Agent lifecycle hooks (on_start, on_finish, on_error)
- Memory-based preference tracking
- API integration patterns with retries and fallbacks
- Tool caching for improved performance

**Capabilities:**

- Flight and hotel searches
- Weather information
- Currency conversion
- Local attractions
- Itinerary generation

**Run this example:**

```bash
uv run examples/advanced/travel_assistant.py
```

### Workflow Demo (`advanced/workflow_demo.py`)

Demonstrates FastADK's workflow system.

**Features demonstrated:**

- Sequential and parallel execution
- Dependency management
- Result processing
- Error handling in workflows

**Run this example:**

```bash
uv run examples/advanced/workflow_demo.py
```

### Context Policies Demo (`advanced/context_policies_demo.py`)

Shows different context management policies.

**Features demonstrated:**

- Most recent messages policy
- Summarization policy
- Hybrid policy combining multiple approaches

**Run this example:**

```bash
uv run examples/advanced/context_policies_demo.py
```

### Memory Backends Demo (`advanced/memory_backends_demo.py`)

Demonstrates different memory backend options.

**Features demonstrated:**

- In-memory storage
- Redis-based persistent storage
- Vector store for semantic search

**Run this example:**

```bash
uv run examples/advanced/memory_backends_demo.py
```

### Multi-Provider Reasoning (`advanced/multi_provider_reasoning.py`)

Demonstrates using multiple LLM providers for reasoning tasks.

**Features demonstrated:**

- Provider switching
- Comparing responses
- Consensus building

**Run this example:**

```bash
uv run examples/advanced/multi_provider_reasoning.py
```

### Observability Demo (`advanced/observability_demo.py`)

Shows how to implement observability in FastADK agents.

**Features demonstrated:**

- Structured logging
- Metrics collection
- Tracing
- Error tracking

**Run this example:**

```bash
uv run examples/advanced/observability_demo.py
```

### Plugin System Demo (`advanced/plugin_system_demo.py`)

Demonstrates the FastADK plugin system.

**Features demonstrated:**

- Creating plugins
- Plugin registration
- Plugin lifecycle hooks
- Custom plugin functionality

**Run this example:**

```bash
uv run examples/advanced/plugin_system_demo.py
```

## UI Examples

### Streamlit Chat App (`ui/streamlit_chat_app.py`)

A simple chat interface built with Streamlit.

**Features demonstrated:**

- Web UI integration
- Session state management
- Asynchronous processing
- Token usage tracking
- Chat history display

**Run this example:**

```bash
uv add streamlit
uv run -m streamlit run examples/ui/streamlit_chat_app.py
```

## API Examples

### HTTP Agent (`api/http_agent.py`)

Demonstrates how to expose an agent via a FastAPI interface.

**Features demonstrated:**

- REST API creation
- Streaming responses
- Request validation
- API documentation

**Run this example:**

```bash
uv run examples/api/http_agent.py
```

## Pattern Examples

### Tool Patterns (`patterns/tool_patterns.py`)

Demonstrates various patterns for implementing tools.

**Features demonstrated:**

- Tool decorators
- Tool error handling
- Tool caching
- Tool validation

**Run this example:**

```bash
uv run examples/patterns/tool_patterns.py
```

### Configuration Patterns (`patterns/configuration_patterns.py`)

Shows different ways to configure FastADK agents.

**Features demonstrated:**

- Environment variables
- Configuration files
- Runtime configuration
- Configuration validation

**Run this example:**

```bash
uv run examples/patterns/configuration_patterns.py
```

## Training Examples

### Fine Tuning Example (`training/fine_tuning_example.py`)

Demonstrates how to fine-tune models with FastADK.

**Features demonstrated:**

- Dataset preparation
- Training configuration
- Model evaluation
- Deployment of fine-tuned models

**Run this example:**

```bash
uv run examples/training/fine_tuning_example.py
```

## Prerequisites

These examples may require additional dependencies. Here are some common ones:

```bash
# Basic dependencies
uv add fastadk

# For advanced examples
uv add redis faiss-cpu httpx python-dotenv

# For UI examples
uv add streamlit

# For API examples
uv add fastapi uvicorn
```

## Notes

- Set appropriate API keys in your environment before running examples
- Some examples use mock data and simulated API calls for demonstration purposes
- When using these examples as a starting point, implement proper error handling and testing in your production code
