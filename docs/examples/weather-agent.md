# Weather Agent Example

This example demonstrates how to create a simple weather agent using FastADK.

## Basic Weather Agent

The following example shows a basic weather agent that can fetch current weather and forecast data.

```python
# weather_agent.py
import requests
from fastadk.core import Agent, BaseAgent, tool
from fastadk.core.exceptions import ToolError

@Agent(
    model="gemini-2.0-pro",
    description="Weather assistant that provides forecasts and recommendations"
)
class WeatherAgent(BaseAgent):
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or "demo_key"  # Use a real API key in production
        
    @tool
    def get_weather(self, city: str) -> dict:
        """Fetch current weather for a city."""
        try:
            # In a real implementation, call a weather API
            # For this example, we'll return mock data
            return {
                "city": city,
                "temperature": "22°C",
                "condition": "sunny",
                "humidity": "45%",
                "wind": "5 km/h"
            }
        except Exception as e:
            raise ToolError(f"Failed to get weather data: {str(e)}")
    
    @tool(cache_ttl=300)  # Cache results for 5 minutes
    def get_forecast(self, city: str, days: int = 3) -> list:
        """Get weather forecast for multiple days."""
        try:
            # In a real implementation, call a weather API
            # For this example, we'll return mock data
            return [
                {"day": 1, "condition": "sunny", "temp": "25°C"},
                {"day": 2, "condition": "cloudy", "temp": "22°C"},
                {"day": 3, "condition": "rainy", "temp": "19°C"},
                {"day": 4, "condition": "stormy", "temp": "17°C"},
                {"day": 5, "condition": "sunny", "temp": "23°C"}
            ][:days]
        except Exception as e:
            raise ToolError(f"Failed to get forecast data: {str(e)}")
    
    @tool
    def get_weather_recommendation(self, condition: str, temperature: str) -> str:
        """Get clothing and activity recommendations based on weather."""
        temp_value = float(temperature.replace("°C", ""))
        
        if temp_value > 25:
            clothing = "light clothing like shorts and t-shirts"
        elif temp_value > 15:
            clothing = "medium-weight clothing like jeans and a light jacket"
        else:
            clothing = "warm clothing including a coat and scarf"
            
        if condition.lower() == "sunny":
            activities = "outdoor activities like hiking or visiting a park"
            additional = "Don't forget sunscreen!"
        elif condition.lower() in ["rainy", "stormy"]:
            activities = "indoor activities like museums or movies"
            additional = "An umbrella would be useful!"
        else:
            activities = "both indoor and outdoor activities"
            additional = ""
            
        return f"I recommend wearing {clothing}. It's a good day for {activities}. {additional}"
```

## Running the Weather Agent

You can run this agent using the FastADK CLI:

```bash
fastadk run weather_agent.py
```

## Example Conversations

Here are some example interactions with the weather agent:

### Getting Current Weather

**User**: What's the weather in Paris today?

**Agent**: I've checked the current weather in Paris for you:
- Temperature: 22°C
- Condition: sunny
- Humidity: 45%
- Wind: 5 km/h

It's a beautiful day in Paris! Would you like a clothing recommendation or forecast for the coming days?

### Getting a Forecast

**User**: Can I get a 5-day forecast for Tokyo?

**Agent**: Here's your 5-day forecast for Tokyo:

Day 1: Sunny with a temperature of 25°C
Day 2: Cloudy with a temperature of 22°C
Day 3: Rainy with a temperature of 19°C
Day 4: Stormy with a temperature of 17°C
Day 5: Sunny with a temperature of 23°C

Would you like any recommendations based on this forecast?

### Getting Recommendations

**User**: What should I wear today if it's sunny and 28°C?

**Agent**: I recommend wearing light clothing like shorts and t-shirts. It's a good day for outdoor activities like hiking or visiting a park. Don't forget sunscreen!

## Adding Real Weather API Integration

To make this a real weather agent, you would integrate with a weather API like OpenWeatherMap:

```python
@tool
def get_weather(self, city: str) -> dict:
    """Fetch current weather for a city."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={self.api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": city,
            "temperature": f"{data['main']['temp']}°C",
            "condition": data['weather'][0]['description'],
            "humidity": f"{data['main']['humidity']}%",
            "wind": f"{data['wind']['speed']} km/h"
        }
    except Exception as e:
        raise ToolError(f"Failed to get weather data: {str(e)}")
```

## Adding Memory

You can enhance the agent with memory to remember user preferences:

```python
@Agent(
    model="gemini-2.0-pro",
    description="Weather assistant with memory",
    memory_backend="redis"
)
class WeatherAgentWithMemory(BaseAgent):
    # ... same tools as before ...
    
    @tool
    def remember_preferred_units(self, temperature_unit: str) -> str:
        """Remember the user's preferred temperature unit (C or F)."""
        if temperature_unit.upper() not in ["C", "F"]:
            raise ToolError("Temperature unit must be C or F")
            
        self.context.set("preferred_unit", temperature_unit.upper())
        return f"I'll remember that you prefer temperatures in °{temperature_unit.upper()}"
    
    @tool
    def get_weather_with_preference(self, city: str) -> dict:
        """Get weather using the user's preferred unit if set."""
        # Get base weather data
        weather = self.get_weather(city)
        
        # Check for user preferences
        preferred_unit = self.context.get("preferred_unit", "C")
        
        # Convert if needed
        if preferred_unit == "F" and "°C" in weather["temperature"]:
            celsius = float(weather["temperature"].replace("°C", ""))
            fahrenheit = (celsius * 9/5) + 32
            weather["temperature"] = f"{fahrenheit:.1f}°F"
            
        return weather
```