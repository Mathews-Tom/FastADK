"""
Chain of Thought Reasoning Demo for FastADK.

This example demonstrates how to create an agent that shows its reasoning process,
making tool selection and thought process transparent to users.

Usage:
    1. Run this example: `uv run examples/basic/reasoning_demo.py`

No external API keys are required as this example uses the simulated provider.
"""

import asyncio
import logging
import random

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@Agent(
    model="gemini-2.5-flash",
    description="An agent that demonstrates reasoning and tool selection process",
    provider="gemini",  # Using Gemini model
    show_reasoning=True,  # Enable reasoning display in the response
)
class ReasoningAgent(BaseAgent):
    """
    A agent that demonstrates how to make the reasoning and tool selection process
    visible to users.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize a database of facts
        self.fact_db = {
            "earth": [
                "Earth is the third planet from the Sun",
                "Earth's radius is approximately 6,371 kilometers",
                "Earth is approximately 4.54 billion years old",
                "71% of Earth's surface is covered by water",
                "Earth has one natural satellite, the Moon",
            ],
            "mars": [
                "Mars is the fourth planet from the Sun",
                "Mars has a thin atmosphere composed primarily of carbon dioxide",
                "Mars has two small moons, Phobos and Deimos",
                "Mars is often called the 'Red Planet'",
                "Mars has the largest volcano in the solar system, Olympus Mons",
            ],
            "jupiter": [
                "Jupiter is the fifth planet from the Sun",
                "Jupiter is the largest planet in our solar system",
                "Jupiter has a strong magnetic field",
                "Jupiter has at least 79 moons",
                "Jupiter has a Giant Red Spot, which is a storm larger than Earth",
            ],
        }
        # Initialize a database of math problems for demo purposes
        self.math_problems = [
            {"question": "What is 25 × 13?", "answer": 325},
            {"question": "If x + 7 = 15, what is x?", "answer": 8},
            {
                "question": "What is the area of a rectangle with length 12 and width 5?",
                "answer": 60,
            },
            {"question": "What is 40% of 85?", "answer": 34},
        ]

    @tool
    def get_planet_facts(self, planet: str) -> dict:
        """
        Get facts about a specific planet.

        Args:
            planet: The name of the planet (earth, mars, jupiter)

        Returns:
            Dictionary with facts about the planet
        """
        planet = planet.lower()
        if planet not in self.fact_db:
            return {
                "planet": planet,
                "found": False,
                "message": f"No facts available for {planet}. Available planets: {', '.join(self.fact_db.keys())}",
            }
        return {
            "planet": planet,
            "found": True,
            "facts": self.fact_db[planet],
        }

    @tool
    def solve_math_problem(self, question: str) -> dict:
        """
        Solve a mathematical problem.

        Args:
            question: The math question to solve

        Returns:
            Dictionary with the solution
        """
        # In a real implementation, this would use an algorithm to solve
        # For demo, we just check if the question matches our database
        for problem in self.math_problems:
            if problem["question"].lower() == question.lower():
                return {
                    "question": question,
                    "answer": problem["answer"],
                    "method": "Direct computation",
                }
        # For questions not in our database, provide a simulated partial solution
        return {
            "question": question,
            "partial_solution": "To solve this problem, we would need to...",
            "note": "This is a placeholder response. In a real implementation, this would compute the actual answer.",
        }

    @tool
    def check_weather(self, location: str, unit: str = "celsius") -> dict:
        """
        Check the weather for a location.

        Args:
            location: The city or location to check weather for
            unit: Temperature unit, either 'celsius' or 'fahrenheit'

        Returns:
            Dictionary with weather information
        """
        # This is a mock implementation for demo purposes
        weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
        # Generate a repeatable but seemingly random temperature based on location
        # This makes the same location always return the same temperature in the demo
        location_hash = sum(ord(c) for c in location) % 30
        temp_c = location_hash + random.randint(-3, 3)
        temp_f = temp_c * 9 / 5 + 32
        return {
            "location": location,
            "temperature": round(temp_f if unit.lower() == "fahrenheit" else temp_c),
            "unit": "°F" if unit.lower() == "fahrenheit" else "°C",
            "condition": weather_conditions[location_hash % len(weather_conditions)],
            "humidity": random.randint(30, 90),
            "note": "This is simulated weather data for demonstration purposes.",
        }

    @tool
    def search_database(self, query: str, max_results: int = 3) -> dict:
        """
        Search a database for information.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results
        """
        # This is a mock implementation that searches through our fact database
        results = []
        query = query.lower()
        # Search through all planet facts
        for planet, facts in self.fact_db.items():
            for fact in facts:
                if query in fact.lower() or query in planet.lower():
                    results.append({"source": planet, "fact": fact})
        # Return results, limited by max_results
        return {
            "query": query,
            "total_matches": len(results),
            "results": results[:max_results],
            "note": "This is a simulated database search for demonstration purposes.",
        }


async def demonstrate_reasoning(agent: ReasoningAgent, query: str) -> None:
    """Run the agent with a query and print the response with reasoning."""
    print(f"\n💬 User Query: {query}")
    print("\n🤖 Processing...")

    try:
        response = await agent.run(query)

        # Display the final response
        print("\n📝 Final Response:")
        print(response)

        # Display token usage if available
        token_stats = agent.get_token_usage_stats()
        if token_stats.get("token_tracking_enabled", False) is not False:
            print("\n📊 Token Usage:")
            print(
                f"  - Session tokens: {token_stats.get('session_tokens_used', 'N/A')}"
            )

        # Display tools used
        if hasattr(agent, "tools_used") and agent.tools_used:
            print("\n🔧 Tools Used:")
            for used_tool in agent.tools_used:
                print(f"  - {used_tool}")
    except (ValueError, RuntimeError, KeyError, AttributeError) as e:
        print(f"\n❌ Error: {e}")


async def main() -> None:
    """Run the reasoning demo with various examples."""
    # Print banner
    print("\n" + "=" * 50)
    print("🧠 FastADK Chain of Thought Reasoning Demo 🧠")
    print("=" * 50)

    # Check for API key
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will still work, but will use simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    try:
        agent = ReasoningAgent()
        # Demonstrate different types of queries that require reasoning
        queries = [
            "Can you tell me some facts about Mars?",
            "What's the weather like in Tokyo and should I bring an umbrella?",
            "I need to solve this math problem: What is 25 × 13?",
            "Search for information about moons in our solar system",
            "Compare Earth and Jupiter based on their size and composition",
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n\n📌 Example {i}/{len(queries)}")
            await demonstrate_reasoning(agent, query)

        print("\n" + "=" * 50)
        print("🏁 Reasoning Demo completed!")
        print("=" * 50)
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Error initializing agent: {e}")
        print("\nIf you're seeing API key errors, please set your GEMINI_API_KEY")
        print("environment variable to use the full functionality of this demo.")


if __name__ == "__main__":
    asyncio.run(main())
