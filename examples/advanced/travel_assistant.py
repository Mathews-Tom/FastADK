"""
End-to-End Travel Assistant Example for FastADK

This example demonstrates a comprehensive travel assistant that showcases all the
key features of FastADK including:

1. Agent configuration with proper decorators
2. Multiple tool implementations with various features (caching, retries, timeouts)
3. Memory usage for conversation context
4. API integrations with error handling
5. Multiple provider support
6. Lifecycle hooks for tracking and monitoring
7. Error handling and fallback mechanisms
8. Type hints for robust parameter handling

Environment Setup:
------------------
1. Install dependencies:
    ```
    uv add httpx python-dotenv
    ```

2. Set up API keys either via environment variables:
    ```
    export GEMINI_API_KEY=your_api_key_here
    # Optional for multi-provider support:
    export OPENAI_API_KEY=your_api_key_here
    export ANTHROPIC_API_KEY=your_api_key_here
    ```

    Or by creating a .env file in the project root:
    ```
    GEMINI_API_KEY=your_api_key_here
    # Optional for multi-provider support:
    OPENAI_API_KEY=your_api_key_here
    ANTHROPIC_API_KEY=your_api_key_here
    ```

3. Run the example:
    ```
    uv run examples/travel_assistant.py
    ```

Note: This example includes mock data and simulated API calls to demonstrate
functionality without requiring actual API access.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, create_app, registry, tool
from fastadk.core.exceptions import ToolError
from fastadk.memory import InMemoryBackend

# Type alias for mypy
JsonDict = dict  # type: ignore

# --- Setup ---
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("travel_assistant")
logging.getLogger("fastadk").setLevel(logging.DEBUG)

# Initialize global memory backend
memory_backend = InMemoryBackend()

# --- Constants and Mock Data ---
COUNTRIES = [
    "France",
    "Japan",
    "Italy",
    "Spain",
    "United States",
    "Australia",
    "Brazil",
    "Canada",
    "Thailand",
    "Mexico",
    "United Kingdom",
    "India",
]

MOCK_FLIGHTS = {
    "Paris": {
        "destinations": ["Tokyo", "Rome", "New York", "London", "Madrid"],
        "avg_price": 450,
    },
    "Tokyo": {
        "destinations": ["Paris", "Los Angeles", "Sydney", "Bangkok"],
        "avg_price": 750,
    },
    "New York": {
        "destinations": ["London", "Paris", "Tokyo", "Mexico City"],
        "avg_price": 500,
    },
    "London": {
        "destinations": ["New York", "Paris", "Tokyo", "Sydney"],
        "avg_price": 400,
    },
    "Rome": {
        "destinations": ["Paris", "Athens", "Barcelona", "Cairo"],
        "avg_price": 200,
    },
    "Sydney": {
        "destinations": ["Tokyo", "Los Angeles", "Auckland", "Singapore"],
        "avg_price": 800,
    },
}

MOCK_HOTELS = {
    "Paris": ["Grand Hotel Paris", "Eiffel Tower View", "Seine River Suites"],
    "Tokyo": ["Tokyo Palace Hotel", "Shinjuku Ryokan", "Imperial Gardens Inn"],
    "New York": ["Manhattan Suites", "Central Park Hotel", "Times Square Lodge"],
    "London": ["Westminster Hotel", "Thames View Inn", "Royal Kensington Hotel"],
    "Rome": ["Colosseum Luxury Suites", "Vatican View Hotel", "Roman Empire Resort"],
    "Sydney": ["Harbor View Hotel", "Opera House Inn", "Bondi Beach Resort"],
}

MOCK_ATTRACTIONS = {
    "Paris": [
        "Eiffel Tower",
        "Louvre Museum",
        "Notre-Dame Cathedral",
        "Arc de Triomphe",
    ],
    "Tokyo": ["Tokyo Skytree", "Sensō-ji Temple", "Meiji Shrine", "Tokyo Disneyland"],
    "New York": [
        "Statue of Liberty",
        "Times Square",
        "Central Park",
        "Empire State Building",
    ],
    "London": ["Tower of London", "British Museum", "Buckingham Palace", "London Eye"],
    "Rome": ["Colosseum", "Vatican City", "Trevi Fountain", "Roman Forum"],
    "Sydney": [
        "Sydney Opera House",
        "Sydney Harbour Bridge",
        "Bondi Beach",
        "Taronga Zoo",
    ],
}

# --- Helper Functions ---


async def fetch_with_fallback(url: str, retries: int = 3, timeout: float = 5.0) -> dict:
    """
    Fetch data from URL with retries and fallback to mock data.

    Args:
        url: The URL to fetch
        retries: Number of retries on failure
        timeout: Timeout in seconds

    Returns:
        Response data as a dictionary
    """
    # For demo purposes, we'll simulate API failures randomly
    if random.random() < 0.3:  # 30% chance of failure
        logger.warning("Simulated API failure for %s", url)
        if "flight" in url:
            return {"error": "API temporarily unavailable", "mock": True}
        raise httpx.RequestError(
            "Simulated connection error", request=httpx.Request("GET", url)
        )

    # Simulate a real API call with delay
    await asyncio.sleep(0.5)

    # Return mock data based on the URL
    if "weather" in url:
        city = url.split("/")[-1].split("?")[0]
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
        temp = random.randint(5, 35)
        return {
            "city": city,
            "temperature": temp,
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 90),
        }
    elif "currency" in url:
        currencies = {"USD": 1.0, "EUR": 0.85, "GBP": 0.75, "JPY": 110.0, "AUD": 1.3}
        return {
            "rates": currencies,
            "base": "USD",
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    elif "flight" in url:
        origin = url.split("origin=")[1].split("&")[0]
        dest = url.split("destination=")[1].split("&")[0]
        if origin in MOCK_FLIGHTS and dest in MOCK_FLIGHTS[origin]["destinations"]:
            base_price = MOCK_FLIGHTS[origin]["avg_price"]
            return {
                "flights": [
                    {
                        "airline": random.choice(
                            ["AirFast", "SkyWings", "GlobalAir", "OceanicAir"]
                        ),
                        "origin": origin,
                        "destination": dest,
                        "price": base_price + random.randint(-100, 200),
                        "departure": "08:30",
                        "arrival": "11:45",
                        "duration": "3h 15m",
                    },
                    {
                        "airline": random.choice(
                            ["AirFast", "SkyWings", "GlobalAir", "OceanicAir"]
                        ),
                        "origin": origin,
                        "destination": dest,
                        "price": base_price + random.randint(-50, 300),
                        "departure": "14:15",
                        "arrival": "17:40",
                        "duration": "3h 25m",
                    },
                ]
            }
        return {"flights": []}

    # Default fallback
    return {"status": "ok", "message": "This is mock data for demonstration purposes"}


# --- Main Agent Class ---


@Agent(
    model="gemini-2.5-flash",
    description="A comprehensive travel assistant that helps plan trips and provides travel information",
    provider="gemini",
    system_prompt="""You are TravelBuddy, an intelligent travel assistant designed to help users plan their perfect trip.
You have access to tools for flight search, weather information, currency conversion, and local recommendations.
Always be helpful, accurate, and consider the user's preferences when providing travel advice.
When using tools, interpret the results and provide them in a friendly, conversational manner.
Remember details from the conversation to provide personalized recommendations.""",
)
class TravelAssistant(BaseAgent):
    """
    TravelBuddy: A comprehensive travel assistant that helps with trip planning,
    flight booking, hotel recommendations, and local information.
    """

    def __init__(self) -> None:
        """Initialize the travel assistant with memory and preferences."""
        super().__init__()
        self.memory = memory_backend
        self.session_start_time = time.time()
        self.start_time = 0
        self.user_preferences: dict[str, Any] = {}
        logger.info("Travel Assistant initialized")

    async def _save_to_memory(
        self, key: str, data: Any, ttl_seconds: int = 3600
    ) -> None:
        """Save data to memory with the given key and TTL."""
        await self.memory.set(key, data, ttl_seconds)
        logger.debug("Saved %s to memory with TTL %ds", key, ttl_seconds)

    async def _get_from_memory(self, key: str) -> Any:
        """Retrieve data from memory by key."""
        entry = await self.memory.get(key)
        if entry:
            return entry.data
        return None

    # --- Weather Tools ---

    @tool  # type: ignore
    async def get_weather(self, city: str) -> dict:
        """
        Get current weather information for a city.

        Args:
            city: The name of the city to get weather for

        Returns:
            Weather data including temperature and conditions
        """
        try:
            # In a real implementation, this would call a weather API
            url = f"https://api.example.com/weather/{city}?units=metric"
            weather_data = await fetch_with_fallback(url)

            # Store the city in user preferences
            user_prefs = await self._get_from_memory("user_preferences") or {}
            if "visited_cities" not in user_prefs:
                user_prefs["visited_cities"] = []
                visited_cities = user_prefs.get("visited_cities", [])
                if isinstance(visited_cities, list) and city not in visited_cities:
                    visited_cities.append(city)
                    user_prefs["visited_cities"] = visited_cities
            await self._save_to_memory("user_preferences", user_prefs)

            return weather_data
        except Exception as e:
            logger.error("Error getting weather for %s: %s", city, e)
            return {
                "city": city,
                "error": "Could not retrieve weather data",
                "fallback": True,
                "temperature": random.randint(15, 25),
                "condition": "unknown",
            }

    # --- Currency Tools ---

    @tool(cache_ttl=86400)  # Cache for 24 hours  # type: ignore
    async def convert_currency(
        self, amount: float, from_currency: str, to_currency: str
    ) -> dict:
        """
        Convert an amount from one currency to another.

        Args:
            amount: The amount to convert
            from_currency: Source currency code (e.g., USD, EUR)
            to_currency: Target currency code (e.g., JPY, GBP)

        Returns:
            Conversion result with rate information
        """
        try:
            url = "https://api.example.com/currency/latest"
            rates_data = await fetch_with_fallback(url)

            # Standardize currency codes
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()

            if "rates" not in rates_data:
                raise ValueError("Invalid rates data")

            # Get conversion rates (handle USD as base)
            rates = rates_data.get("rates", {})
            if from_currency == "USD":
                from_rate = 1.0
            else:
                from_rate = rates.get(from_currency)

            to_rate = rates.get(to_currency)

            if not from_rate or not to_rate:
                raise ValueError(f"Rate not found for {from_currency} or {to_currency}")

            # Calculate converted amount
            conversion_rate = to_rate / from_rate
            converted_amount = amount * conversion_rate

            return {
                "original": {"amount": amount, "currency": from_currency},
                "converted": {
                    "amount": round(converted_amount, 2),
                    "currency": to_currency,
                },
                "rate": round(conversion_rate, 4),
                "date": rates_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            }
        except Exception as e:
            logger.error("Currency conversion error: %s", e)
            raise ToolError(f"Could not convert currency: {e}")

    # --- Flight Tools ---

    @tool(timeout=10, retries=2)  # Longer timeout with retries  # type: ignore
    async def search_flights(self, origin: str, destination: str, date: str) -> dict:
        """
        Search for flights between two cities on a specific date.

        Args:
            origin: Departure city
            destination: Arrival city
            date: Travel date in YYYY-MM-DD format

        Returns:
            List of available flights with prices and times
        """
        try:
            # Validate date format
            try:
                parsed_date = datetime.strptime(date, "%Y-%m-%d")
                # Check if date is in the past
                if parsed_date.date() < datetime.now().date():
                    return {"error": "Travel date cannot be in the past", "flights": []}
            except ValueError:
                return {
                    "error": "Invalid date format. Please use YYYY-MM-DD format",
                    "flights": [],
                }

            # In a real implementation, this would call a flight search API
            url = f"https://api.example.com/flights/search?origin={origin}&destination={destination}&date={date}"
            flights_data = await fetch_with_fallback(url)

            # Save search to memory for recommendations
            searches = await self._get_from_memory("flight_searches") or []
            # Add the search to our search history
            if isinstance(searches, list):
                searches.append(
                    {
                        "origin": str(origin),
                        "destination": str(destination),
                        "date": str(date),
                    }
                )
            await self._save_to_memory("flight_searches", searches)

            # Add a helpful message if no flights found
            if "flights" in flights_data and len(flights_data["flights"]) == 0:
                flights_data["message"] = (
                    f"No flights found from {origin} to {destination} on {date}"
                )

            return flights_data
        except Exception as e:
            logger.error("Error searching flights: %s", e)
            return {
                "error": f"Flight search failed: {str(e)}",
                "flights": [],
                "fallback": True,
            }

    # --- Hotel and Accommodation Tools ---

    @tool
    async def find_hotels(
        self, city: str, check_in: str, check_out: str, budget: str | None = None
    ) -> dict:
        """
        Find hotels in a city for a specific date range.

        Args:
            city: The city to search for hotels
            check_in: Check-in date in YYYY-MM-DD format
            check_out: Check-out date in YYYY-MM-DD format
            budget: Optional budget constraint (low, medium, high)

        Returns:
            List of available hotels with prices and ratings
        """
        try:
            # Validate city
            if city not in MOCK_HOTELS:
                return {
                    "city": city,
                    "hotels": [],
                    "message": f"No hotels found in {city}",
                }

            # Get hotels and add ratings and prices
            hotels = []
            for hotel_name in MOCK_HOTELS[city]:
                base_price = float(
                    100 + (COUNTRIES.index(city) if city in COUNTRIES else 5) * 20
                )

                # Adjust price based on budget preference
                if budget:
                    if budget.lower() == "low":
                        base_price = base_price * 0.7
                    elif budget.lower() == "high":
                        base_price = base_price * 1.5

                hotels.append(
                    {
                        "name": hotel_name,
                        "rating": round(random.uniform(3.0, 5.0), 1),
                        "price_per_night": round(base_price + random.randint(-20, 50)),
                        "amenities": random.sample(
                            [
                                "WiFi",
                                "Pool",
                                "Spa",
                                "Restaurant",
                                "Gym",
                                "Bar",
                                "Airport Shuttle",
                            ],
                            k=random.randint(3, 5),
                        ),
                    }
                )

            # Sort by rating descending, with proper type handling
            def get_rating(hotel: dict) -> float:
                try:
                    return float(hotel.get("rating", 0.0))
                except (ValueError, TypeError):
                    return 0.0

            hotels.sort(key=get_rating, reverse=True)

            return {
                "city": city,
                "check_in": check_in,
                "check_out": check_out,
                "hotels": hotels,
            }
        except Exception as e:
            logger.error("Error finding hotels: %s", e)
            return {"city": city, "hotels": [], "error": str(e)}

    # --- Local Information Tools ---

    @tool(cache_ttl=604800)  # Cache for 1 week  # type: ignore
    async def get_attractions(self, city: str) -> dict:
        """
        Get popular attractions and landmarks in a city.

        Args:
            city: The city to get attractions for

        Returns:
            List of attractions with descriptions
        """
        if city not in MOCK_ATTRACTIONS:
            return {
                "city": city,
                "attractions": [],
                "message": f"No attraction information found for {city}",
            }

        attractions = []
        for attraction in MOCK_ATTRACTIONS[city]:
            attractions.append(
                {
                    "name": attraction,
                    "rating": round(random.uniform(3.5, 5.0), 1),
                    "category": random.choice(
                        [
                            "Historical",
                            "Cultural",
                            "Natural",
                            "Entertainment",
                            "Architecture",
                        ]
                    ),
                    "suggested_duration": f"{random.randint(1, 5)} hours",
                }
            )

        return {"city": city, "attractions": attractions}

    @tool
    async def get_travel_advisory(self, country: str) -> dict:
        """
        Get travel advisory and safety information for a country.

        Args:
            country: The country to get advisory for

        Returns:
            Travel advisory information including safety level
        """
        # In a real implementation, this would call a travel advisory API
        safety_levels = ["Low Risk", "Medium Risk", "Exercise Caution", "High Risk"]

        # Simulate data
        return {
            "country": country,
            "safety_level": random.choice(safety_levels),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "advisory_details": f"Travelers to {country} should follow standard safety precautions.",
            "entry_requirements": "Valid passport required. Visa requirements may apply.",
        }

    # --- User Preference Tools ---

    @tool
    async def save_preference(self, category: str, preference: str) -> dict:
        """
        Save a user preference for future recommendations.

        Args:
            category: Category of preference (e.g., food, climate, budget)
            preference: The user's preference for this category

        Returns:
            Confirmation of the saved preference
        """
        user_prefs = await self._get_from_memory("user_preferences") or {}

        if category == "reset":
            await self._save_to_memory("user_preferences", {})
            return {"message": "All preferences have been reset"}

        user_prefs[category] = preference
        await self._save_to_memory("user_preferences", user_prefs)

        return {
            "message": f"Your {category} preference has been saved as '{preference}'",
            "all_preferences": user_prefs,
        }

    @tool
    async def get_preferences(self) -> dict:
        """
        Retrieve all saved user preferences.

        Returns:
            All stored user preferences
        """
        user_prefs = await self._get_from_memory("user_preferences") or {}

        if not user_prefs:
            return {"message": "No preferences saved yet"}

        return {"preferences": user_prefs}

    # --- Recommendation Tools ---

    @tool
    async def recommend_destination(self) -> dict:
        """
        Recommend destinations based on user preferences and search history.

        Returns:
            List of recommended destinations with rationale
        """
        user_prefs = await self._get_from_memory("user_preferences") or {}
        flight_searches = await self._get_from_memory("flight_searches") or []

        # Get cities the user has shown interest in
        interested_cities = []
        if flight_searches:
            for search in flight_searches:
                if search["destination"] not in interested_cities:
                    interested_cities.append(search["destination"])
                if search["origin"] not in interested_cities:
                    interested_cities.append(search["origin"])

        if "visited_cities" in user_prefs:
            for city in user_prefs["visited_cities"]:
                if city not in interested_cities:
                    interested_cities.append(city)

        # If we don't have enough information, recommend popular destinations
        if not interested_cities or len(interested_cities) < 2:
            recommendations = random.sample(
                list(MOCK_ATTRACTIONS.keys()), min(3, len(MOCK_ATTRACTIONS))
            )
            return {
                "recommendations": [
                    {
                        "destination": city,
                        "attractions": MOCK_ATTRACTIONS[city][:2],
                        "rationale": "Popular destination with many attractions",
                    }
                    for city in recommendations
                ],
                "based_on": "Popular destinations worldwide",
            }

        # Use preferences to recommend destinations
        climate_pref = user_prefs.get("climate", "").lower()
        budget_pref = user_prefs.get("budget", "").lower()

        recommendations = []
        potential_cities = list(set(MOCK_ATTRACTIONS.keys()) - set(interested_cities))

        if not potential_cities:
            potential_cities = list(MOCK_ATTRACTIONS.keys())

        random.shuffle(potential_cities)
        for city in potential_cities[:3]:
            rationale = f"Based on your interest in {', '.join(interested_cities[:2])}"

            if climate_pref:
                rationale += f" and preference for {climate_pref} climate"

            if budget_pref:
                rationale += f" with {budget_pref} budget"

            recommendations.append(
                {
                    "destination": city,
                    "attractions": MOCK_ATTRACTIONS[city][:2],
                    "rationale": rationale,
                }
            )

        return {
            "recommendations": recommendations,
            "based_on": "Your preferences and search history",
        }

    # --- Helper and Utility Tools ---

    @tool
    async def generate_itinerary(self, destination: str, days: int) -> dict:
        """
        Generate a day-by-day travel itinerary for a destination.

        Args:
            destination: The destination city
            days: Number of days for the itinerary (1-7)

        Returns:
            Day-by-day itinerary plan
        """
        days = min(max(1, int(days)), 7)  # Ensure days is between 1 and 7

        if destination not in MOCK_ATTRACTIONS:
            return {
                "destination": destination,
                "error": f"No information available for {destination}",
            }

        attractions = MOCK_ATTRACTIONS[destination].copy()
        restaurants = [
            f"{destination} Gourmet Restaurant",
            f"Traditional {destination} Cuisine",
            f"{destination} Street Food Market",
            f"Local {destination} Bistro",
            f"Waterfront Dining in {destination}",
        ]

        itinerary = {}
        for day in range(1, days + 1):
            # Generate different activities for each day
            morning = (
                random.choice(attractions) if attractions else "Free time to explore"
            )
            if morning in attractions:
                attractions.remove(morning)

            # If we run out of attractions, add generic activities
            if not attractions:
                attractions = [
                    f"Explore {destination} neighborhoods",
                    f"Visit local markets in {destination}",
                    f"{destination} walking tour",
                    f"Cultural experience in {destination}",
                ]

            afternoon = random.choice(attractions)
            attractions.remove(afternoon)

            evening = random.choice(restaurants)
            restaurants.remove(evening)
            if not restaurants:
                restaurants = [f"Dinner in {destination}"]

            itinerary[f"Day {day}"] = {
                "morning": morning,
                "afternoon": afternoon,
                "evening": evening,
                "note": f"Optional: {random.choice(['Local tour', 'Shopping', 'Museum visit', 'Relaxation'])}",
            }

        return {"destination": destination, "days": days, "itinerary": itinerary}

    # --- Lifecycle Hooks ---

    def on_start(self) -> None:
        """Hook called when the agent starts processing a request."""
        self.start_time = time.time()
        self.tools_used: list[str] = []
        logger.info("Starting processing request")

        # We can't use await directly in non-async methods,
        # so we'll create a task to check preferences
        asyncio.create_task(self._check_preferences())

    async def _check_preferences(self) -> None:
        """Check if we have user preferences already."""
        user_prefs = await self._get_from_memory("user_preferences")
        if user_prefs:
            logger.info("User has %d saved preferences", len(user_prefs))
            # Update user preferences in memory for future use
            self.user_preferences = user_prefs

    def on_finish(self, result: str) -> None:
        """Hook called when the agent finishes processing a request."""
        processing_time = time.time() - self.start_time
        logger.info(
            "Request processed in %.2fs using %d tools",
            processing_time,
            len(self.tools_used),
        )
        logger.info(
            "Tools used: %s", ", ".join(self.tools_used) if self.tools_used else "None"
        )

        # Analyze response length
        response_words = len(result.split())
        logger.info("Response length: %d words", response_words)

        # Create task to update metrics asynchronously
        asyncio.create_task(self._update_metrics(processing_time, len(self.tools_used)))

    async def _update_metrics(
        self, processing_time: float, tools_used_count: int
    ) -> None:
        """Update session metrics asynchronously."""
        session_metrics = await self._get_from_memory("session_metrics") or {
            "requests_count": 0,
            "avg_processing_time": 0,
            "total_tools_used": 0,
        }

        # Update metrics
        current_count = session_metrics["requests_count"]
        new_count = current_count + 1

        # Calculate new average
        old_avg = session_metrics["avg_processing_time"]
        new_avg = (old_avg * current_count + processing_time) / new_count

        session_metrics["requests_count"] = int(new_count)
        session_metrics["avg_processing_time"] = float(new_avg)
        session_metrics["total_tools_used"] = int(
            session_metrics.get("total_tools_used", 0) + tools_used_count
        )

        await self._save_to_memory("session_metrics", session_metrics)

    def on_error(self, error: Exception) -> None:
        """Hook called when the agent encounters an error."""
        logger.error("Error during processing: %s", str(error))

        # Record error asynchronously
        asyncio.create_task(self._record_error(error))

    async def _record_error(self, error: Exception) -> None:
        """Record error for analysis."""
        errors = await self._get_from_memory("errors") or []
        errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "error": str(error),
                "type": type(error).__name__,
            }
        )
        await self._save_to_memory("errors", errors)


# --- Simple CLI Interface ---


async def demo_travel_assistant() -> None:
    """Run a demo of the travel assistant with a simple command-line interface."""
    agent = TravelAssistant()
    print("\n" + "=" * 50)
    print("  🌍  TRAVEL ASSISTANT DEMO  ✈️")
    print("=" * 50)
    print("\nWelcome to the Travel Assistant demo! Ask about flights, weather,")
    print("hotels, attractions, or get travel recommendations.")
    print("\nType 'exit' to quit the demo.")
    print("\nExample queries:")
    print("- What's the weather like in Tokyo?")
    print("- Find flights from Paris to Rome on 2025-08-15")
    print("- I need hotel recommendations in London")
    print("- What are the top attractions in Sydney?")
    print("- Generate a 3-day itinerary for New York")
    print("- Convert 100 USD to EUR")
    print("=" * 50 + "\n")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit", "bye"):
            break

        print("\nProcessing your request...")
        try:
            response = await agent.run(user_input)
            print(f"\nTravel Assistant: {response}")
        except Exception as e:
            print(f"\nTravel Assistant: I'm sorry, I encountered an error: {str(e)}")
            logger.exception("Error during agent execution")

    print("\nThank you for using the Travel Assistant demo! Goodbye.")


# --- HTTP API Server ---


def create_http_server() -> Any:
    """Create and run the FastAPI server with the Travel Assistant."""
    # Register the agent with the registry
    registry.register(TravelAssistant)

    # Create and return the FastAPI app
    return create_app()


# --- Main Function ---


async def test_run() -> None:
    """Run a simple test of the travel assistant with a predefined query."""
    agent = TravelAssistant()
    print("\n=== Testing Travel Assistant with direct tool calls ===\n")

    # Test tools directly to demonstrate they work
    print("Testing get_weather tool...")
    weather = await agent.get_weather("Tokyo")
    print(f"Weather in Tokyo: {weather}")

    print("\nTesting get_attractions tool...")
    attractions = await agent.get_attractions("Paris")
    print(f"Attractions in Paris: {attractions}")

    print("\nTesting currency conversion...")
    conversion = await agent.convert_currency(100, "USD", "EUR")
    print(f"Currency conversion: {conversion}")

    print("\nTesting itinerary generation...")
    itinerary = await agent.generate_itinerary("New York", 2)
    print(f"Itinerary for New York: {itinerary}")

    print("\n=== Testing complete agent with user queries ===\n")

    # Now test the agent with actual queries to see if it routes to tools
    test_queries = [
        "I want to find a hotel in Rome for August 15-20",
        "Tell me about the weather in Tokyo",
        "What attractions should I visit in Paris?",
    ]

    for query in test_queries:
        print(f"\nTest Query: {query}")
        try:
            response = await agent.run(query)
            print(f"Response: {response[:150]}...(truncated)")
            print(
                f"Tools used: {', '.join(agent.tools_used) if agent.tools_used else 'None'}"
            )
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\n=== Test completed ===")


if __name__ == "__main__":
    # By default, run the test function
    asyncio.run(test_run())

    # To run the interactive CLI demo instead, uncomment this:
    # asyncio.run(demo_travel_assistant())

    # To run the HTTP API server, uncomment these:
    # import uvicorn
    # app = create_http_server()
    # uvicorn.run(app, host="127.0.0.1", port=8000)
