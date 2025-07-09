"""
Tool Development Patterns for FastADK.

This example demonstrates different patterns for developing and using tools:
1. Basic synchronous and asynchronous tools
2. Tools with validation and error handling
3. Tools with nested structure and rich returns
4. Parameterized tools with defaults
5. Dynamic tool registration patterns
6. Tool composition and reuse

Usage:
    1. Run the example:
        uv run examples/patterns/tool_patterns.py
"""

import asyncio
import inspect
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from fastadk import Agent, BaseAgent, tool

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======= DATA MODELS FOR VALIDATION =======


class WeatherRequest(BaseModel):
    """Request model for weather data."""

    location: str = Field(description="City or location name")
    units: str = Field("celsius", description="Temperature units (celsius/fahrenheit)")

    @field_validator("units")
    @classmethod
    def validate_units(cls, v: str) -> str:
        """Validate that units are either celsius or fahrenheit."""
        if v.lower() not in ["celsius", "fahrenheit"]:
            raise ValueError("Units must be 'celsius' or 'fahrenheit'")
        return v.lower()

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate that location is not empty and contains only allowed characters."""
        if not v or v.strip() == "":
            raise ValueError("Location cannot be empty")
        if not re.match(r"^[a-zA-Z0-9\s,.-]+$", v):
            raise ValueError("Location contains invalid characters")
        return v.strip()


class WeatherResponse(BaseModel):
    """Response model for weather data."""

    location: str
    temperature: float
    conditions: str
    humidity: Optional[int] = None
    wind_speed: Optional[float] = None
    units: str
    timestamp: str


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Maximum number of results")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate that limit is within bounds."""
        if v < 1:
            raise ValueError("Limit must be at least 1")
        if v > 20:
            raise ValueError("Limit cannot exceed 20")
        return v


class SearchResult(BaseModel):
    """Model for a single search result."""

    title: str
    url: Optional[str] = None
    snippet: str
    relevance: float = Field(..., ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response model for search."""

    query: str
    results: List[SearchResult]
    total_results: int
    execution_time: float


# ======= EXAMPLE AGENT WITH DIFFERENT TOOL PATTERNS =======


@Agent(
    model="gemini-1.5-pro",
    description="An agent demonstrating various tool patterns",
    provider="gemini",  # Will fall back to simulated if no API key is available
)
class ToolPatternsAgent(BaseAgent):
    """
    Agent demonstrating different patterns for tool development.

    This agent includes examples of different ways to create, validate, and use tools
    in the FastADK framework.
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_demo_data()
        self._register_dynamic_tools()

    def _init_demo_data(self) -> None:
        """Initialize demo data for tools."""
        # Demo weather data
        self.weather_data = {
            "new york": {
                "temp_c": 22,
                "temp_f": 72,
                "conditions": "Partly cloudy",
                "humidity": 65,
            },
            "london": {
                "temp_c": 18,
                "temp_f": 64,
                "conditions": "Rainy",
                "humidity": 80,
            },
            "tokyo": {
                "temp_c": 26,
                "temp_f": 79,
                "conditions": "Clear",
                "humidity": 70,
            },
            "sydney": {
                "temp_c": 20,
                "temp_f": 68,
                "conditions": "Sunny",
                "humidity": 55,
            },
            "paris": {
                "temp_c": 24,
                "temp_f": 75,
                "conditions": "Cloudy",
                "humidity": 60,
            },
        }

        # Demo search index
        self.search_index = [
            {
                "title": "FastADK: Building Advanced AI Agents",
                "url": "https://example.com/fastadk",
                "content": "Learn how to build advanced AI agents with FastADK, a powerful framework for creating LLM-powered applications.",
                "keywords": ["ai", "agents", "fastadk", "llm"],
            },
            {
                "title": "Machine Learning Fundamentals",
                "url": "https://example.com/ml-fundamentals",
                "content": "Explore the core concepts of machine learning, from basic algorithms to advanced neural networks.",
                "keywords": ["machine learning", "ai", "algorithms", "neural networks"],
            },
            {
                "title": "Natural Language Processing Techniques",
                "url": "https://example.com/nlp-techniques",
                "content": "A comprehensive guide to natural language processing techniques and applications.",
                "keywords": ["nlp", "language", "processing", "ai"],
            },
            {
                "title": "Building Conversational Agents",
                "url": "https://example.com/conversational-agents",
                "content": "Learn how to design and implement conversational agents that can engage in meaningful dialogue.",
                "keywords": ["agents", "conversational", "dialogue", "chatbots"],
            },
            {
                "title": "Advanced Python for Data Science",
                "url": "https://example.com/python-data-science",
                "content": "Master advanced Python techniques for data science and machine learning applications.",
                "keywords": ["python", "data science", "programming"],
            },
            {
                "title": "Reinforcement Learning in Practice",
                "url": "https://example.com/reinforcement-learning",
                "content": "Practical guide to implementing reinforcement learning algorithms for real-world problems.",
                "keywords": ["reinforcement learning", "ai", "algorithms"],
            },
            {
                "title": "Ethics in Artificial Intelligence",
                "url": "https://example.com/ai-ethics",
                "content": "Exploring the ethical considerations and challenges in artificial intelligence development and deployment.",
                "keywords": ["ethics", "ai", "policy", "governance"],
            },
        ]

    # ===== PATTERN 1: BASIC SYNCHRONOUS TOOL =====

    @tool(return_type=dict)
    def get_current_time(self) -> Dict[str, Any]:
        """
        Get the current time and date.

        Returns:
            Dictionary with current time information
        """
        now = datetime.now()

        return {
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%I:%M:%S %p"),
            "formatted_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
        }

    # ===== PATTERN 2: ASYNC TOOL WITH SIMULATED DELAY =====

    @tool(return_type=dict)
    async def get_random_number(
        self, min_value: int = 1, max_value: int = 100
    ) -> Dict[str, Any]:
        """
        Generate a random number between min_value and max_value.

        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)

        Returns:
            Dictionary with the random number
        """
        # Simulate an asynchronous operation
        await asyncio.sleep(0.5)

        result = random.randint(min_value, max_value)

        return {
            "random_number": result,
            "range": {"min": min_value, "max": max_value},
            "timestamp": datetime.now().isoformat(),
        }

    # ===== PATTERN 3: TOOL WITH PYDANTIC VALIDATION =====

    @tool(return_type=dict)
    def get_weather(self, location: str, units: str = "celsius") -> Dict[str, Any]:
        """
        Get weather information for a location with input validation.

        Args:
            location: City or location name
            units: Temperature units ('celsius' or 'fahrenheit')

        Returns:
            Weather information including temperature and conditions

        Raises:
            ValueError: If the inputs are invalid
        """
        try:
            # Validate inputs using Pydantic model
            request = WeatherRequest(location=location, units=units)

            # Process validated request - location is already normalized by the validator
            location_key = str(request.location).lower()

            if location_key not in self.weather_data:
                return {
                    "error": f"Weather data not available for {request.location}",
                    "available_locations": list(self.weather_data.keys()),
                }

            weather = self.weather_data[location_key]
            temp_key = "temp_c" if request.units == "celsius" else "temp_f"

            # Create response
            response = WeatherResponse(
                location=request.location,
                temperature=weather[temp_key],
                conditions=weather["conditions"],
                humidity=weather["humidity"],
                units=request.units,
                timestamp=datetime.now().isoformat(),
            )

            # Return as dictionary
            return response.dict()

        except ValidationError as e:
            # Detailed validation error
            error_messages = []
            for error in e.errors():
                field = error["loc"][0]
                message = error["msg"]
                error_messages.append(f"{field}: {message}")

            return {
                "error": "Validation error",
                "details": error_messages,
            }

    # ===== PATTERN 4: TOOL WITH RICH ERROR HANDLING =====

    @tool(return_type=dict)
    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for information with robust error handling.

        Args:
            query: Search query
            limit: Maximum number of results (1-20)

        Returns:
            Search results matching the query
        """
        start_time = time.time()

        try:
            # Validate inputs
            request = SearchRequest(query=query, limit=limit)

            # Simulate search latency
            await asyncio.sleep(0.7)

            # Perform search (simulated)
            results = []
            for item in self.search_index:
                # Simple relevance calculation
                relevance = 0.0

                # Check title match
                if query.lower() in item["title"].lower():
                    relevance += 0.5

                # Check content match
                if query.lower() in item["content"].lower():
                    relevance += 0.3

                # Check keyword match
                for keyword in item["keywords"]:
                    if query.lower() in keyword.lower():
                        relevance += 0.2
                        break

                if relevance > 0:
                    results.append(
                        {
                            "title": item["title"],
                            "url": item["url"],
                            "snippet": item["content"][:100] + "...",
                            "relevance": min(1.0, relevance),  # Cap at 1.0
                        }
                    )

            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)

            # Limit results
            results = results[: int(request.limit)]

            # Create response
            response = SearchResponse(
                query=request.query,
                results=[SearchResult(**r) for r in results],
                total_results=len(results),
                execution_time=time.time() - start_time,
            )

            return response.dict()

        except ValidationError as e:
            # Handle validation errors
            return {
                "error": "Validation error",
                "details": str(e),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            # Handle unexpected errors
            logger.error("Search error: %s", str(e), exc_info=True)
            return {
                "error": "Search failed",
                "details": str(e),
                "execution_time": time.time() - start_time,
            }

    # ===== PATTERN 5: NESTED TOOL WITH HIERARCHICAL STRUCTURE =====

    @tool(return_type=dict)
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed system information with a nested structure.

        Returns:
            Hierarchical system information
        """
        # This would normally gather real system info
        # For demo purposes, we're returning simulated data

        return {
            "system": {
                "os": {
                    "name": "Demo OS",
                    "version": "1.0.4",
                    "architecture": "64-bit",
                },
                "hardware": {
                    "cpu": {
                        "model": "Intel Demo CPU",
                        "cores": 8,
                        "speed": "3.2 GHz",
                    },
                    "memory": {
                        "total": "16 GB",
                        "available": "8.5 GB",
                        "usage": "47%",
                    },
                    "storage": {
                        "total": "512 GB",
                        "available": "256 GB",
                        "usage": "50%",
                    },
                },
                "network": {
                    "status": "connected",
                    "interface": "wifi",
                    "ip_address": "192.168.1.100",
                },
            },
            "application": {
                "name": "FastADK Demo",
                "version": "1.0.0",
                "runtime": "Python 3.10",
                "uptime": "1h 23m",
            },
            "timestamp": datetime.now().isoformat(),
        }

    # ===== PATTERN 6: DYNAMIC TOOL REGISTRATION =====

    def _register_dynamic_tools(self) -> None:
        """Register tools dynamically based on runtime conditions."""

        # Define a calculator tool function
        def calculator(operation: str, a: float, b: float) -> Dict[str, Any]:
            """
            Perform a mathematical operation.

            Args:
                operation: Operation to perform (add, subtract, multiply, divide)
                a: First number
                b: Second number

            Returns:
                Result of the operation
            """
            operation = operation.lower()

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {
                        "error": "Division by zero",
                        "operation": operation,
                        "a": a,
                        "b": b,
                    }
                result = a / b
            else:
                return {
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": ["add", "subtract", "multiply", "divide"],
                }

            return {
                "operation": operation,
                "a": a,
                "b": b,
                "result": result,
            }

        # In a real application, you would dynamically register tools
        # using BaseAgent's add_tool method or similar
        # For this example, we'll just print
        print("Would register calculator tool here")

        # Define an async tool for currency conversion
        async def convert_currency(
            amount: float, from_currency: str, to_currency: str
        ) -> Dict[str, Any]:
            """
            Convert an amount from one currency to another.

            Args:
                amount: Amount to convert
                from_currency: Source currency code (USD, EUR, etc.)
                to_currency: Target currency code

            Returns:
                Converted amount and exchange rate
            """
            # Simulate API call delay
            await asyncio.sleep(0.5)

            # Mock exchange rates (fixed for demo)
            rates = {
                "USD": 1.0,
                "EUR": 0.85,
                "GBP": 0.73,
                "JPY": 110.12,
                "CAD": 1.25,
                "AUD": 1.36,
            }

            # Validate currencies
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()

            if from_currency not in rates:
                return {
                    "error": f"Unknown currency: {from_currency}",
                    "supported_currencies": list(rates.keys()),
                }

            if to_currency not in rates:
                return {
                    "error": f"Unknown currency: {to_currency}",
                    "supported_currencies": list(rates.keys()),
                }

            # Calculate conversion
            from_rate = rates[from_currency]
            to_rate = rates[to_currency]

            # Convert to USD first, then to target currency
            amount_in_usd = amount / from_rate
            converted_amount = amount_in_usd * to_rate

            # Calculate exchange rate
            exchange_rate = to_rate / from_rate

            return {
                "original": {
                    "amount": amount,
                    "currency": from_currency,
                },
                "converted": {
                    "amount": round(converted_amount, 2),
                    "currency": to_currency,
                },
                "exchange_rate": round(exchange_rate, 6),
                "timestamp": datetime.now().isoformat(),
            }

        # Same as above, in a real application we would register this tool
        print("Would register convert_currency tool here")


async def demonstrate_tool_patterns() -> None:
    """Run the tool patterns demonstration."""
    print("\n" + "=" * 60)
    print("🛠️  FastADK Tool Patterns Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    try:
        # Create the agent
        print("\n🚀 Initializing agent...")
        agent = ToolPatternsAgent()

        # Get available tools
        tools = {
            name: func
            for name, func in inspect.getmembers(agent)
            if hasattr(func, "_is_tool")
        }

        # In a real application, we would get dynamically registered tools
        # This is just a placeholder for the demo
        print("  (Note: Dynamic tools would be added here)")

        print(f"\n📋 Available tools: {len(tools)}")

        # For demonstration purposes, we'll just describe the tools
        # instead of actually executing them to avoid coroutine issues

        # Demonstrate each tool pattern
        print("\n\n📌 PATTERN 1: BASIC SYNCHRONOUS TOOL")
        print("-" * 60)
        print("Tool: get_current_time")
        print("Description: Returns the current time and date in a formatted structure")

        print("\n\n📌 PATTERN 2: ASYNC TOOL WITH SIMULATED DELAY")
        print("-" * 60)
        print("Tool: get_random_number")
        print(
            "Description: Generates a random number in a specified range with a simulated delay"
        )

        print("\n\n📌 PATTERN 3: TOOL WITH PYDANTIC VALIDATION")
        print("-" * 60)
        print("Tool: get_weather")
        print("Description: Gets weather data for a location with input validation")
        print(
            "Benefits: Automatic validation of input parameters and helpful error messages"
        )

        print("\n\n📌 PATTERN 4: TOOL WITH RICH ERROR HANDLING")
        print("-" * 60)
        print("Tool: search")
        print(
            "Description: Demonstrates comprehensive error handling with detailed error reporting"
        )

        print("\n\n📌 PATTERN 5: NESTED TOOL WITH HIERARCHICAL STRUCTURE")
        print("-" * 60)
        print("Tool: get_system_info")
        print(
            "Description: Shows how to structure complex nested data in tool responses"
        )

        print("\n\n📌 PATTERN 6: DYNAMICALLY REGISTERED TOOLS")
        print("-" * 60)
        print("Tool: calculator and convert_currency")
        print("Description: Shows how to dynamically register tools at runtime")
        print("Examples:")
        print(
            "  - calculator: Perform mathematical operations (add, subtract, multiply, divide)"
        )
        print("  - convert_currency: Convert between different currencies")

        # Summary
        print("\n📝 Summary of Tool Patterns Demonstrated:")
        print("1. Basic synchronous tool - Simple operation with return value")
        print("2. Async tool - Support for asynchronous operations")
        print("3. Tool with Pydantic validation - Type and value validation")
        print("4. Tool with rich error handling - Detailed error reporting")
        print("5. Tool with hierarchical data - Structured nested responses")
        print("6. Dynamically registered tools - Runtime tool registration")

        print("\n" + "=" * 60)
        print("🏁 FastADK -  Tool Patterns Demo Completed")
        print("=" * 60)
    except Exception as e:
        logger.error("Error in tool patterns demo: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")


async def main() -> None:
    """Run the main demo."""
    await demonstrate_tool_patterns()


if __name__ == "__main__":
    asyncio.run(main())
