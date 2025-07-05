"""
Example HTTP Agent for FastADK.

This example demonstrates serving multiple agents via HTTP API.
It shows how to:
1. Create multiple agent classes with different providers
2. Register them with the FastADK registry
3. Create a FastAPI application that serves all agents
4. Run the server with uvicorn

To run this example, you need:
1. Install requirements: `uv add fastapi uvicorn python-dotenv`
2. Set up API keys for the LLM providers you want to use, either via:

    Environment variables:
    ```
    export GEMINI_API_KEY=your_key_here
    export OPENAI_API_KEY=your_key_here
    export ANTHROPIC_API_KEY=your_key_here
    ```

    Or by creating a .env file in the project root:
    ```
    GEMINI_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here
    ANTHROPIC_API_KEY=your_key_here
    ```

3. Run the server: `uv run http_agent.py`
4. Access the API documentation at http://127.0.0.1:8000/docs
"""

import random
from datetime import datetime

from fastadk import Agent, BaseAgent, create_app, registry, tool


@Agent(
    model="gemini-1.5-pro",
    description="An assistant that can provide weather information and facts",
    provider="gemini",
)
class WeatherAssistant(BaseAgent):
    """A demo agent that can provide weather information and fun facts."""

    def __init__(self) -> None:
        super().__init__()
        # Store some conversation context
        self.favorite_cities: list[str] = []

    @tool
    def get_weather(self, city: str) -> dict:
        """
        Get the current weather for a city.

        Args:
            city: The name of the city to get weather for

        Returns:
            Weather information including temperature and conditions
        """
        # In a real implementation, this would call a weather API
        # For demo purposes, we'll return mock data
        conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
        temp = random.randint(0, 35)

        # Add to favorite cities if not already there
        if city not in self.favorite_cities:
            self.favorite_cities.append(city)

        return {
            "city": city,
            "temperature": f"{temp}°C",
            "condition": random.choice(conditions),
            "timestamp": datetime.now().isoformat(),
        }

    @tool(cache_ttl=300)  # Cache results for 5 minutes
    def get_forecast(self, city: str, days: int = 3) -> list:
        """
        Get a weather forecast for a city.

        Args:
            city: The name of the city to get forecast for
            days: Number of days to forecast (default: 3)

        Returns:
            A list of daily forecasts
        """
        conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy"]
        forecast = []

        # Use city parameter to generate the forecast
        base_temp = (
            hash(city) % 10 + 15
        )  # Generate a base temperature based on city name

        for i in range(days):
            temp_high = base_temp + random.randint(0, 10)
            temp_low = base_temp - random.randint(0, 10)

            forecast.append(
                {
                    "day": (datetime.now().day + i) % 30 + 1,
                    "condition": random.choice(conditions),
                    "high_temp": f"{temp_high}°C",
                    "low_temp": f"{temp_low}°C",
                    "precipitation": f"{random.randint(0, 100)}%",
                }
            )

        return forecast

    @tool
    def get_fun_fact(self, topic: str | None = None) -> str:
        """
        Get a random fun fact about a topic.

        Args:
            topic: Optional topic for the fun fact

        Returns:
            A fun fact as a string
        """
        # Use the topic if provided, otherwise use a default
        if topic is None:
            topic = random.choice(["weather", "climate", "science"])

        facts = {
            "weather": [
                "Lightning strikes the Earth about 8.6 million times per day.",
                "The world's highest recorded temperature is 134°F (56.7°C) in Death Valley, USA.",
                "A hurricane can release energy equivalent to 10 atomic bombs.",
            ],
            "climate": [
                "The Earth's average temperature has increased by about 1°C in the past century.",
                "The oceans absorb about 30% of CO2 emissions.",
                "Arctic sea ice is declining at a rate of 13.1% per decade.",
            ],
            "science": [
                "There are more atoms in a glass of water than glasses of water in all the oceans combined.",
                "Honeybees can recognize human faces.",
                "A day on Venus is longer than a year on Venus.",
            ],
        }

        # Handle cases where the topic isn't in our database
        if topic not in facts:
            return f"I don't have facts about {topic}, but did you know that octopuses have three hearts?"

        return random.choice(facts[topic])

    @tool
    def list_favorite_cities(self) -> list:
        """
        List all cities that have been checked for weather in this session.

        Returns:
            A list of city names
        """
        return self.favorite_cities

    def on_start(self) -> None:
        """Called when the agent starts processing a request."""
        self.tools_used = []

    def on_finish(self, result: str) -> None:
        """Called when the agent finishes processing a request."""
        print(f"Agent completed processing with {len(self.tools_used)} tools used.")


@Agent(model="gpt-4", description="A mathematical assistant", provider="openai")
class MathHelper(BaseAgent):
    """An agent that helps with mathematical calculations."""

    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @tool
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b

    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    @tool
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    @tool
    def square_root(self, x: float) -> float:
        """Calculate the square root of a number."""
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return x**0.5

    @tool
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base**exponent


@Agent(
    model="claude-3-haiku-20240307",
    description="An assistant that helps with text-related tasks",
    provider="anthropic",
)
class TextHelper(BaseAgent):
    """An agent that helps with text analysis and generation."""

    @tool
    def count_words(self, text: str) -> int:
        """Count the number of words in a text."""
        return len(text.split())

    @tool
    def count_characters(self, text: str, include_spaces: bool = True) -> int:
        """
        Count the number of characters in a text.

        Args:
            text: The text to analyze
            include_spaces: Whether to include spaces in the count

        Returns:
            Number of characters
        """
        if include_spaces:
            return len(text)
        return len(text.replace(" ", ""))

    @tool
    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        Generate a summary of the given text.

        Args:
            text: The text to summarize
            max_length: Maximum length of the summary

        Returns:
            A summary of the text
        """
        # In a real implementation, this would call the model
        # For demo, we'll just truncate the text
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @tool
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text: The text to analyze

        Returns:
            The detected language
        """
        # Simple mock implementation
        common_english = ["the", "and", "is", "in", "to", "you", "that", "it"]
        common_spanish = ["el", "la", "es", "en", "y", "de", "que", "por"]
        common_french = ["le", "la", "est", "en", "et", "de", "que", "pour"]

        words = text.lower().split()

        english_count = sum(1 for word in words if word in common_english)
        spanish_count = sum(1 for word in words if word in common_spanish)
        french_count = sum(1 for word in words if word in common_french)

        if english_count > spanish_count and english_count > french_count:
            return "English"
        elif spanish_count > english_count and spanish_count > french_count:
            return "Spanish"
        elif french_count > english_count and french_count > spanish_count:
            return "French"
        else:
            return "Unknown"


if __name__ == "__main__":
    # Example of how to start the API server programmatically
    import uvicorn

    # Register all agent classes
    registry.register(WeatherAssistant)
    registry.register(MathHelper)
    registry.register(TextHelper)

    # Create and run the FastAPI app
    app = create_app()
    print("\n🚀 Starting FastADK API Server")
    print("============================")
    print("Available Agents:")
    print("- WeatherAssistant (gemini-1.5-pro)")
    print("- MathHelper (gpt-4)")
    print("- TextHelper (claude-3-haiku)")
    print("\nAPI documentation available at http://127.0.0.1:8000/docs")
    print("============================")
    uvicorn.run(app, host="127.0.0.1", port=8000)
