"""
An example of a live, fully-featured weather agent built with FastADK.

This agent demonstrates:
- Using the @Agent and @tool decorators.
- A live integration with the Gemini provider and a real-world API (wttr.in).
- Asynchronous tool implementation using `httpx`.
- Loading a system prompt from an external text file.
- Type hinting for robust tool parameter handling.
- Docstrings for automatic tool descriptions and help text.

To run this example, you need:
1. Install httpx: `uv add httpx`
2. Set up your Gemini API key either by:
    - Setting an environment variable: `export GEMINI_API_KEY=your_api_key_here`
    - Or creating a .env file in the project root with: `GEMINI_API_KEY=your_api_key_here`
        (requires python-dotenv: `uv add python-dotenv`)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import httpx

from fastadk import Agent, BaseAgent, tool

# --- Setup ---
# Configure logging for better output during execution.
# Setting the fastadk logger to DEBUG provides detailed framework-level insights.
logging.basicConfig(level=logging.INFO)
logging.getLogger("fastadk").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Helper Function for API Calls ---


async def _get_weather_data(city: str) -> dict | None:
    """
    A helper function to fetch weather data from the free wttr.in API.

    Args:
        city: The city to fetch weather data for.

    Returns:
        A dictionary containing the JSON response, or None if an error occurs.
    """
    url = f"https://wttr.in/{city}?format=j1"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=10.0)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error for %s: %s", city, e.response.status_code)
            return None
        except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as e:
            logger.error("Error fetching or parsing weather for %s: %s", city, e)
            return None


# --- Agent Definition ---

# Use pathlib to create a robust path to the system prompt file.
# This ensures that the script can be run from any directory.
prompt_path = Path(__file__).parent / "weather_agent_prompt.txt"


@Agent(
    model="gemini-2.5-flash",
    description="A professional meteorologist agent using live data",
    system_prompt=prompt_path,
    provider="gemini",
)
class WeatherAgent(BaseAgent):
    """
    This agent provides current weather conditions and multi-day forecasts
    for any city in the world, acting as a professional meteorologist.
    """

    @tool
    async def get_current_weather(self, city: str) -> dict:
        """
        Get the current weather for a specific city.

        :param city: The city name, e.g., 'San Francisco', 'Tokyo', 'London'.
        """
        data = await _get_weather_data(city)
        if not data or "current_condition" not in data:
            return {"error": f"Could not retrieve weather for {city}."}

        current = data["current_condition"][0]
        return {
            "city": data["nearest_area"][0]["areaName"][0]["value"],
            "temp_C": current["temp_C"],
            "temp_F": current["temp_F"],
            "feels_like_C": current["FeelsLikeC"],
            "feels_like_F": current["FeelsLikeF"],
            "condition": current["weatherDesc"][0]["value"],
            "humidity": current["humidity"],
        }

    @tool
    async def get_weather_forecast(self, city: str, days: int = 3) -> list:
        """
        Get the weather forecast for a city.

        :param city: The city name, e.g., 'Paris', 'New York'.
        :param days: The number of days for the forecast (must be between 1 and 3).
        """
        # The model might pass a float, so we robustly cast it to an integer.
        try:
            days = int(days)
            if not 1 <= days <= 3:
                return [{"error": "Number of forecast days must be between 1 and 3."}]
        except (ValueError, TypeError):
            return [{"error": "The 'days' parameter must be a valid integer."}]

        data = await _get_weather_data(city)
        if not data or "weather" not in data:
            return [{"error": f"Could not retrieve forecast for {city}."}]

        forecasts = []
        for day_data in data["weather"][:days]:
            try:
                date = datetime.strptime(day_data["date"], "%Y-%m-%d").strftime(
                    "%A, %b %d"
                )
                forecasts.append(
                    {
                        "date": date,
                        "avg_temp_C": day_data["avgtempC"],
                        "avg_temp_F": day_data["avgtempF"],
                        "condition": day_data["hourly"][4]["weatherDesc"][0]["value"],
                    }
                )
            except (KeyError, IndexError) as e:
                logger.warning("Could not parse part of the forecast data: %s", e)

        return forecasts

    # --- Lifecycle Hooks (Optional) ---

    def on_start(self) -> None:
        """This hook is called before the agent processes the input."""
        logger.info("WeatherAgent LIVE processing starting")

    def on_finish(self, result: str) -> None:
        """This hook is called after the agent generates a final response."""
        logger.info("WeatherAgent LIVE response length: %d", len(result))


if __name__ == "__main__":
    # This block allows for direct execution of the script for quick testing.
    # It demonstrates how to instantiate and run the agent programmatically.
    import os

    async def test_agent():
        """Test the WeatherAgent with a sample query about London weather."""
        # Check if GEMINI_API_KEY environment variable is set
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n⚠️  Warning: GEMINI_API_KEY environment variable not found.")
            print("   The agent might not be able to generate proper responses.")
            print("   Set it with: export GEMINI_API_KEY=your_api_key_here\n")

        try:
            agent = WeatherAgent()
            print("\n--- Testing Agent with a sample query ---")
            response = await agent.run("What's the weather like in London?")
            print(f"\nFinal Response:\n{response}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"\n❌ Error: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Make sure your GEMINI_API_KEY is set and valid")
            print("2. Check your internet connection (needed to access wttr.in)")
            print("3. Ensure httpx is installed: uv add httpx")

    asyncio.run(test_agent())
