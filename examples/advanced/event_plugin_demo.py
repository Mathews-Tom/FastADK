"""
Event Plugin System Demo for FastADK.

This example demonstrates the new event-based plugin system:
1. Creating event-based plugins
2. Registering plugins with agents
3. Handling various agent lifecycle events
4. Monitoring agent activity through plugins

Usage:
    1. Run the example:
        uv run examples/advanced/event_plugin_demo.py
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from logging_plugin import LoggingPlugin

from fastadk import Agent, BaseAgent, tool
from fastadk.core.plugin import Plugin, supports_plugin

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetricsPlugin(Plugin):
    """A plugin that collects and reports metrics on agent performance."""

    def __init__(self) -> None:
        """Initialize the plugin."""
        super().__init__(
            name="metrics_plugin",
            version="1.0.0",
            description="Collects and reports metrics on agent performance",
        )
        self.logger = logging.getLogger("fastadk.plugins.metrics")
        self.tool_timings: Dict[str, List[float]] = {}
        self.llm_tokens: Dict[str, int] = {}
        self.start_time = time.time()

    async def initialize(
        self, plugin_manager: Any = None, **kwargs
    ) -> None:  # pylint: disable=unused-argument
        """Initialize the plugin."""
        await super().initialize(plugin_manager, **kwargs)
        self.logger.info("Initializing %s v%s", self.name, self.version)
        self.register_event_handlers()

    def register_event_handlers(self) -> None:
        """Register event handlers for this plugin."""
        self.register_event_handler("tool:called", self.on_tool_called)
        self.register_event_handler("tool:completed", self.on_tool_completed)
        self.register_event_handler("llm:response", self.on_llm_response)

    async def on_tool_called(self, event_data: Dict[str, Any]) -> None:
        """Record tool start time."""
        # Store the start time in event_data for later use
        event_data["_metrics_start_time"] = time.time()

    async def on_tool_completed(self, event_data: Dict[str, Any]) -> None:
        """Record tool timing."""
        tool_name = event_data.get("tool_name", "unknown")
        duration = event_data.get("duration", 0)

        if tool_name not in self.tool_timings:
            self.tool_timings[tool_name] = []

        self.tool_timings[tool_name].append(duration)

    async def on_llm_response(self, event_data: Dict[str, Any]) -> None:
        """Record token usage."""
        model = event_data.get("model", "unknown")
        tokens = event_data.get("tokens", 0)

        if model not in self.llm_tokens:
            self.llm_tokens[model] = 0

        self.llm_tokens[model] += tokens

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generate a metrics report."""
        # Calculate tool timing statistics
        tool_stats = {}
        for tool_name, timings in self.tool_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                min_time = min(timings)
                max_time = max(timings)

                tool_stats[tool_name] = {
                    "calls": len(timings),
                    "avg_duration": round(avg_time, 3),
                    "min_duration": round(min_time, 3),
                    "max_duration": round(max_time, 3),
                }

        # Calculate token usage
        token_usage = {model: tokens for model, tokens in self.llm_tokens.items()}

        # Calculate overall statistics
        total_calls = sum(len(timings) for timings in self.tool_timings.values())
        total_tokens = sum(self.llm_tokens.values())
        elapsed_time = time.time() - self.start_time

        return {
            "elapsed_time": round(elapsed_time, 2),
            "total_tool_calls": total_calls,
            "total_tokens": total_tokens,
            "tool_statistics": tool_stats,
            "token_usage": token_usage,
        }

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated."""
        self.logger.info("Shutting down %s", self.name)


# Mark the agent as supporting our plugins
@supports_plugin(LoggingPlugin)
@supports_plugin(MetricsPlugin)
@Agent(
    model="gemini-1.5-pro",
    description="An agent that demonstrates the event plugin system",
    provider="gemini",  # Will fall back to simulated if no API key is available
)
class EventDemoAgent(BaseAgent):
    """Agent demonstrating the event plugin system."""

    @tool
    def get_weather(self, location: str) -> dict:
        """
        Get weather information for a location.

        Args:
            location: The city to get weather for

        Returns:
            Weather information
        """
        # Simulate API call
        import random

        # Simulate network delay
        time.sleep(0.5)

        # Demo data
        weather_data = {
            "new york": {
                "temperature": 72,
                "conditions": "Partly cloudy",
                "humidity": 65,
            },
            "los angeles": {
                "temperature": 85,
                "conditions": "Sunny",
                "humidity": 50,
            },
            "chicago": {
                "temperature": 68,
                "conditions": "Cloudy",
                "humidity": 70,
            },
            "unknown": {
                "temperature": random.randint(60, 90),
                "conditions": random.choice(["Sunny", "Cloudy", "Rainy", "Windy"]),
                "humidity": random.randint(30, 90),
            },
        }

        location = location.lower()
        weather = weather_data.get(location, weather_data["unknown"])

        return {
            "location": location,
            "temperature": weather["temperature"],
            "conditions": weather["conditions"],
            "humidity": weather["humidity"],
        }

    @tool
    def get_metrics_report(self) -> dict:
        """
        Get a metrics report from the metrics plugin.

        Returns:
            Metrics report
        """
        metrics_plugin = None
        for plugin in self._active_plugins.values():
            if plugin.name == "metrics_plugin":
                metrics_plugin = plugin
                break

        if not metrics_plugin:
            return {
                "success": False,
                "message": "Metrics plugin not active",
                "metrics": {},
            }

        report = metrics_plugin.get_metrics_report()

        return {
            "success": True,
            "metrics": report,
        }


async def demonstrate_event_plugin_system() -> None:
    """Run the event plugin system demonstration."""
    print("\n" + "=" * 60)
    print("🔌 FastADK Event Plugin System Demo")
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
        # The plugin system will automatically initialize the supported plugins
        print("\n🚀 Initializing agent with plugins...")
        agent = EventDemoAgent()

        # Demonstrate agent with plugins
        print("\n📋 Running agent with automatic event tracking...")

        # Test the weather tool
        print("\n🌤️  Weather Tool:")
        locations = ["New York", "Chicago", "Mars"]

        for location in locations:
            print(f"\n  Getting weather for {location}...")
            result = await agent.execute_tool("get_weather", location=location)
            print(f"  📍 {location}: {result['temperature']}°F, {result['conditions']}")
            print(f"  🌡️  Humidity: {result['humidity']}%")

        # Get metrics report
        print("\n📈 Metrics Report from Plugin:")
        result = await agent.execute_tool("get_metrics_report")

        if result.get("success", False):
            metrics = result.get("metrics", {})
            print(f"  ⏱️  Elapsed time: {metrics.get('elapsed_time')}s")
            print(f"  🔢 Total tool calls: {metrics.get('total_tool_calls')}")
            print(f"  🔤 Total tokens: {metrics.get('total_tokens')}")

            print("\n  Tool Statistics:")
            tool_stats = metrics.get("tool_statistics", {})
            for tool_name, stats in tool_stats.items():
                print(
                    f"   - {tool_name}: {stats['calls']} calls, avg: {stats['avg_duration']}s"
                )
        else:
            print(f"  ❌ {result.get('message')}")

        # Test agent with a complex prompt
        print("\n💬 Testing agent with a complex prompt...")
        response = await agent.run(
            "What's the weather like in Chicago and New York? Can you compare them?"
        )
        print(f"\nAgent response: {response}")

        # Get updated metrics
        print("\n📊 Updated Metrics Report:")
        result = await agent.execute_tool("get_metrics_report")

        if result.get("success", False):
            metrics = result.get("metrics", {})
            print(f"  ⏱️  Elapsed time: {metrics.get('elapsed_time')}s")
            print(f"  🔢 Total tool calls: {metrics.get('total_tool_calls')}")
            print(f"  🔤 Total tokens: {metrics.get('total_tokens')}")

        # Clean up
        print("\n🧹 Shutting down plugins...")
        await agent.plugin_manager.shutdown_plugins()

        print("\n" + "=" * 60)
        print("🏁 FastADK - Event Plugin System Demo Completed")
        print("=" * 60)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error in event plugin demo: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")


async def main() -> None:
    """Run the main demo."""
    await demonstrate_event_plugin_system()


if __name__ == "__main__":
    asyncio.run(main())
