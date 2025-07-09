# HTTP Agent Example

This example demonstrates how to serve multiple FastADK agents via a HTTP API using FastAPI. It shows how to create, configure, and expose different agent classes with their own tools and capabilities through a unified API interface.

## Features Demonstrated

- Creating multiple agent classes with different LLM providers
- Registering agents with the FastADK registry
- Setting up a FastAPI application that serves all registered agents
- Tool caching with TTL (Time To Live)
- Maintaining agent state between requests
- Using lifecycle hooks for request monitoring
- Documenting API endpoints with OpenAPI

## Prerequisites

To run this example, you need:

1. Install the required dependencies:

   ```bash
   uv add fastapi uvicorn python-dotenv
   ```

2. Set up API keys for the LLM providers you want to use, either via environment variables:

   ```bash
   export GEMINI_API_KEY=your_key_here
   export OPENAI_API_KEY=your_key_here
   export ANTHROPIC_API_KEY=your_key_here
   ```

   Or by creating a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

3. Run the server:

   ```bash
   uv run http_agent.py
   ```

4. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How It Works

This example creates three different agent classes:

1. **WeatherAssistant**: Uses Gemini provider to offer weather information and fun facts
   - Demonstrates stateful tools that remember user preferences
   - Shows tool caching with the `cache_ttl` parameter
   - Implements lifecycle hooks with `on_start` and `on_finish`

2. **MathHelper**: Uses OpenAI provider to perform mathematical calculations
   - Shows basic error handling with division by zero
   - Demonstrates numerical tools with validation

3. **TextHelper**: Uses Anthropic provider to analyze and manipulate text
   - Shows text processing tools with different parameters
   - Demonstrates handling of complex string inputs and outputs

The example uses FastADK's built-in registry and `create_app()` function to automatically expose all registered agents via a FastAPI application.

## API Endpoints

When running, the server exposes the following main endpoints:

- **GET /agents**: List all available agents
- **POST /agents/{agent_name}/run**: Run an agent with a user query
- **GET /agents/{agent_name}/tools**: List all tools available for a specific agent
- **POST /agents/{agent_name}/tools/{tool_name}**: Execute a specific tool
- **OpenAPI documentation**: Available at `/docs` endpoint

## Expected Output

When you run the script, you should see output similar to:

```bash
🚀 Starting FastADK API Server
============================
Available Agents:
- WeatherAssistant (gemini-1.5-pro)
- MathHelper (gpt-4)
- TextHelper (claude-3-haiku)

API documentation available at http://127.0.0.1:8000/docs
============================
INFO:     Started server process [1234]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## Using the API

### Example API Requests

1. **List all agents**:

   ```bash
   curl -X GET http://127.0.0.1:8000/agents
   ```

2. **Run an agent with a query**:

   ```bash
   curl -X POST http://127.0.0.1:8000/agents/WeatherAssistant/run \
     -H "Content-Type: application/json" \
     -d '{"query": "What's the weather like in Paris?"}'
   ```

3. **Execute a specific tool**:

   ```bash
   curl -X POST http://127.0.0.1:8000/agents/MathHelper/tools/multiply \
     -H "Content-Type: application/json" \
     -d '{"a": 5, "b": 7}'
   ```

## Key Concepts

1. **Agent Registry**: FastADK's registry system allows different agent classes to be discovered and exposed via the API.

2. **Multi-Provider Support**: The example shows how to use different LLM providers (Gemini, OpenAI, Anthropic) in the same application.

3. **Tool Caching**: The `cache_ttl` parameter demonstrates how to cache tool results to improve performance and reduce API calls.

4. **Stateful Agents**: The WeatherAssistant demonstrates maintaining state between requests with the `favorite_cities` list.

5. **OpenAPI Integration**: FastADK automatically generates OpenAPI documentation for all registered agents and their tools.

## Best Practices Demonstrated

- Creating specialized agents for different domains (weather, math, text)
- Using appropriate LLM providers for different types of tasks
- Implementing proper error handling in tools
- Caching appropriate tool results to improve performance
- Maintaining minimal state for better user experience
- Using lifecycle hooks for monitoring and debugging
