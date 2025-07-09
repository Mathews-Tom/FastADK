# Weather Agent Example

This example demonstrates a live, fully-featured weather agent built with FastADK. The agent can provide current weather conditions and multi-day forecasts for any city in the world, acting as a professional meteorologist.

## Features Demonstrated

- Using the `@Agent` and `@tool` decorators to create a functional agent
- Live integration with the Gemini provider for AI responses
- Real-world API integration with wttr.in for weather data
- Asynchronous tool implementation using `httpx`
- Loading a system prompt from an external text file
- Type hinting for robust tool parameter handling
- Docstrings for automatic tool descriptions and help text
- Lifecycle hooks for monitoring agent behavior

## Prerequisites

To run this example, you need:

1. Install the required HTTP client:

   ```bash
   uv add httpx
   ```

2. Set up your Gemini API key either by:
   - Setting an environment variable:

     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```

   - Or creating a .env file in the project root with:

     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

     (requires python-dotenv: `uv add python-dotenv`)

## How It Works

This example creates a `WeatherAgent` class that extends `BaseAgent` and uses two main tools:

1. `get_current_weather`: Fetches and processes current weather conditions for a specified city
2. `get_weather_forecast`: Retrieves a multi-day weather forecast for a city

The agent leverages a system prompt from the file `weather_agent_prompt.txt` that instructs it to act as a professional meteorologist named "WxWizard" and to format weather information in a specific way.

The example also demonstrates lifecycle hooks through the `on_start` and `on_finish` methods, which are called before and after agent execution, respectively.

## Expected Output

When you run the script, you should see output similar to:

```bash
INFO:fastadk.agent:Initialized Gemini model gemini-2.5-flash
INFO:fastadk.agent:Initialized agent WeatherAgent with 2 tools

--- Testing Agent with a sample query ---
INFO:__main__:WeatherAgent LIVE processing starting
INFO:__main__:WeatherAgent LIVE response length: 521
INFO:fastadk.agent:Agent execution completed in 4.19s

Final Response:
The current weather in London is cloudy with a temperature of 18°C (64°F).
It feels like 17°C (63°F) due to a light breeze and 75% humidity.

I'd recommend bringing a light jacket or sweater as it might feel a bit cool,
especially if you'll be outside for extended periods. An umbrella isn't
necessary as there's no rain in the immediate forecast.

The forecast for the next three days shows a gradual warming trend with
temperatures rising to 21°C by Friday, with partly cloudy conditions expected.
```

## Code Structure

The script includes:

- A helper function `_get_weather_data` for API calls
- A `WeatherAgent` class with two tool methods
- Lifecycle hook methods
- A test function to demonstrate the agent in action

## Troubleshooting

If you encounter issues:

1. Make sure your GEMINI_API_KEY is set and valid
2. Check your internet connection (needed to access wttr.in)
3. Ensure httpx is installed: `uv add httpx`
