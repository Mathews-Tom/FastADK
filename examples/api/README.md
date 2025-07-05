# API FastADK Examples

This directory contains examples that demonstrate how to serve FastADK agents via API.

## HTTP Agent Example

The `http_agent.py` example demonstrates:

- Creating multiple agent classes with different providers (Gemini, OpenAI, Anthropic)
- Registering agents with the FastADK registry
- Creating a FastAPI application that serves all agents
- Running the server with uvicorn

### Setup and Running

1. Install required dependencies:

   ```bash
   uv add fastapi uvicorn python-dotenv
   ```

2. Set up environment variables or a `.env` file:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_api_key_here
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Run the example:

   ```bash
   uv run http_agent.py
   ```

4. Access the API documentation:
   Open your browser to <http://127.0.0.1:8000/docs>

### Expected Output

```bash
🚀 Starting FastADK API Server
============================
Available Agents:
- WeatherAssistant (gemini-1.5-pro)
- MathHelper (gpt-4)
- TextHelper (claude-3-haiku)

API documentation available at http://127.0.0.1:8000/docs
============================
INFO:     Started server process [9129]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Using the API

Once the server is running, you can interact with the agents using the API:

1. **Chat with an agent**:

   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/agents/WeatherAssistant/chat' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"message": "What is the weather like in Paris?"}'
   ```

   **Example Response**:

   ```json
   {
     "response": "Based on my mock data, the current weather in Paris is cloudy with a temperature of 22°C. The local time is 2025-07-05T21:15:30.123456. \n\nPlease note that in a real implementation, this would fetch live weather data from a weather API. If you need accurate weather information, I recommend checking a dedicated weather service.",
     "conversation_id": "conv_7d8f9a2e3b1c",
     "metadata": {
       "execution_time": 1.23,
       "model": "gemini-1.5-pro",
       "tools_used": ["get_weather"]
     }
   }
   ```

2. **Execute a specific tool**:

   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/agents/MathHelper/tools/add' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"a": 5, "b": 3}'
   ```

   **Example Response**:

   ```json
   {
     "result": 8,
     "metadata": {
       "execution_time": 0.02,
       "tool": "add"
     }
   }
   ```

3. **List available agents**:

   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/agents' \
     -H 'accept: application/json'
   ```

   **Example Response**:

   ```json
   {
     "agents": [
       {
         "name": "WeatherAssistant",
         "description": "An assistant that can provide weather information and facts",
         "model": "gemini-1.5-pro",
         "provider": "gemini",
         "tools": [
           {"name": "get_weather", "description": "Get the current weather for a city"},
           {"name": "get_forecast", "description": "Get a weather forecast for a city"},
           {"name": "get_fun_fact", "description": "Get a random fun fact about a topic"},
           {"name": "list_favorite_cities", "description": "List all cities that have been checked for weather in this session"}
         ]
       },
       {
         "name": "MathHelper",
         "description": "A mathematical assistant",
         "model": "gpt-4",
         "provider": "openai",
         "tools": [
           {"name": "add", "description": "Add two numbers together"},
           {"name": "subtract", "description": "Subtract b from a"},
           {"name": "multiply", "description": "Multiply two numbers together"},
           {"name": "divide", "description": "Divide a by b"},
           {"name": "square_root", "description": "Calculate the square root of a number"},
           {"name": "power", "description": "Raise base to the power of exponent"}
         ]
       },
       {
         "name": "TextHelper",
         "description": "An assistant that helps with text-related tasks",
         "model": "claude-3-haiku-20240307",
         "provider": "anthropic",
         "tools": [
           {"name": "count_words", "description": "Count the number of words in a text"},
           {"name": "count_characters", "description": "Count the number of characters in a text"},
           {"name": "generate_summary", "description": "Generate a summary of the given text"},
           {"name": "detect_language", "description": "Detect the language of the given text"}
         ]
       }
     ]
   }
   ```
