"""
Example weather agent using FastADK.

This example demonstrates a simple agent that provides weather information
using the @Agent and @tool decorators.
"""

import asyncio
import random

from fastadk import Agent, BaseAgent, tool


@Agent(
    model="gemini-1.5-pro",
    description="Weather assistant that provides weather information for cities",
    provider="gemini",
)
class WeatherAgent(BaseAgent):
    """An agent that provides weather information for cities."""

    # City data for demo purposes
    CITIES = {
        "london": {"temp": "18°C", "condition": "Cloudy", "humidity": "75%"},
        "new york": {"temp": "22°C", "condition": "Sunny", "humidity": "60%"},
        "tokyo": {"temp": "26°C", "condition": "Rainy", "humidity": "85%"},
        "paris": {"temp": "20°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "sydney": {"temp": "24°C", "condition": "Sunny", "humidity": "50%"},
    }

    @tool
    def get_weather(self, city: str) -> dict[str, str]:
        """
        Fetch current weather for a city.

        Args:
            city: The name of the city to get weather for

        Returns:
            A dictionary with weather information
        """
        city = city.lower()
        if city in self.CITIES:
            return self.CITIES[city]
        else:
            # Generate random weather for unknown cities
            conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Foggy", "Snowy"]
            return {
                "city": city,
                "temp": f"{random.randint(15, 30)}°C",
                "condition": random.choice(conditions),
                "humidity": f"{random.randint(30, 90)}%",
            }

    @tool(cache_ttl=3600)  # Cache for 1 hour
    def get_forecast(self, city: str, days: int = 3) -> list[dict[str, str]]:
        """
        Get a weather forecast for the specified number of days.

        Args:
            city: The name of the city to get a forecast for
            days: The number of days to forecast (default: 3)

        Returns:
            A list of daily forecasts
        """
        conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Foggy", "Snowy"]
        return [
            {
                "day": f"Day {i+1}",
                "temp": f"{random.randint(15, 30)}°C",
                "condition": random.choice(conditions),
                "humidity": f"{random.randint(30, 90)}%",
            }
            for i in range(min(days, 7))  # Limit to 7 days max
        ]

    @tool
    def get_weather_alerts(self, city: str) -> list[str]:
        """
        Get any active weather alerts for a city.

        Args:
            city: The name of the city to check for alerts

        Returns:
            A list of active weather alerts
        """
        # Randomly return an alert or an empty list
        if random.random() < 0.3:  # 30% chance of an alert
            alerts = [
                "Flood warning in effect until tomorrow morning",
                "High wind advisory for the next 6 hours",
                "Thunderstorm watch in effect",
                "Heat advisory issued for today",
                "Air quality alert - moderate pollution levels",
            ]
            return [random.choice(alerts)]
        return []


async def main():
    """Run the weather agent with example queries."""
    agent = WeatherAgent()

    # Ask about current weather
    query = "What's the weather like in London today?"
    print(f"\nQuery: {query}")
    response = await agent.run(query)
    print(f"Response: {response}")

    # Ask about forecast
    query = "What's the forecast for Tokyo for the next 5 days?"
    print(f"\nQuery: {query}")
    response = await agent.run(query)
    print(f"Response: {response}")

    # Ask about weather alerts
    query = "Are there any weather alerts for New York?"
    print(f"\nQuery: {query}")
    response = await agent.run(query)
    print(f"Response: {response}")

    # See which tools were used
    print(f"\nTools used in the last query: {agent.tools_used}")


if __name__ == "__main__":
    asyncio.run(main())
