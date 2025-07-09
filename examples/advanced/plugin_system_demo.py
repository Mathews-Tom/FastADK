"""
Plugin System Demo for FastADK.

This example demonstrates FastADK's plugin architecture:
1. Creating custom plugins
2. Registering plugins with the framework
3. Managing plugin lifecycle
4. Handling plugin events
5. Extending core functionality

Usage:
    1. Run the example:
        uv run examples/advanced/plugin_system_demo.py
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool


# Define a base Plugin class for this example
class Plugin:
    """Base class for plugins in this example."""

    def __init__(self, name: str, version: str, description: str) -> None:
        """Initialize the plugin.

        Args:
            name: The name of the plugin
            version: The version of the plugin
            description: A description of the plugin
        """
        self.name = name
        self.version = version
        self.description = description

    async def initialize(self, plugin_manager) -> None:
        """Initialize the plugin.

        Args:
            plugin_manager: The plugin manager instance
        """
        pass

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated."""
        pass


# Create a custom PluginManager for our demo
class DemoPluginManager:
    """A simplified plugin manager for the demo."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self.plugins = {}
        self.event_handlers = {}

    async def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin.

        Args:
            plugin: The plugin to register
        """
        self.plugins[plugin.name] = plugin
        await plugin.initialize(self)

    def register_event_handler(self, event_name: str, handler) -> None:
        """Register an event handler.

        Args:
            event_name: The name of the event
            handler: The event handler function
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    async def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Emit an event to all registered handlers.

        Args:
            event_name: The name of the event
            event_data: The event data
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                await handler(event_data)

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin by name.

        Args:
            plugin_name: The name of the plugin

        Returns:
            The plugin if found, None otherwise
        """
        return self.plugins.get(plugin_name)

    async def shutdown(self) -> None:
        """Shut down all plugins."""
        for plugin in self.plugins.values():
            await plugin.shutdown()


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======= CUSTOM PLUGINS =======


class LoggingPlugin(Plugin):
    """A plugin that logs agent activities and events."""

    def __init__(self) -> None:
        super().__init__(
            name="logging_plugin",
            version="1.0.0",
            description="Logs agent activities and events",
        )
        self.logger = logging.getLogger("fastadk.plugins.logging")
        self.events_logged = 0
        self.start_time = time.time()

    async def initialize(self, plugin_manager: DemoPluginManager) -> None:
        """Initialize the plugin."""
        self.logger.info("Initializing %s v%s", self.name, self.version)
        self.register_event_handlers(plugin_manager)

    def register_event_handlers(self, plugin_manager: DemoPluginManager) -> None:
        """Register event handlers for this plugin."""
        plugin_manager.register_event_handler("agent:created", self.on_agent_created)
        plugin_manager.register_event_handler(
            "agent:initialized", self.on_agent_initialized
        )
        plugin_manager.register_event_handler("tool:called", self.on_tool_called)
        plugin_manager.register_event_handler("tool:completed", self.on_tool_completed)
        plugin_manager.register_event_handler("llm:request", self.on_llm_request)
        plugin_manager.register_event_handler("llm:response", self.on_llm_response)

    async def on_agent_created(self, event_data: Dict[str, Any]) -> None:
        """Handle agent creation event."""
        agent_class = event_data.get("agent_class", "Unknown")
        agent_id = id(event_data.get("agent", None))
        self.logger.info("Agent created: %s (id: %s)", agent_class, agent_id)
        self.events_logged += 1

    async def on_agent_initialized(self, event_data: Dict[str, Any]) -> None:
        """Handle agent initialization event."""
        agent = event_data.get("agent", None)
        agent_id = id(agent) if agent else "Unknown"
        self.logger.info("Agent initialized (id: %s)", agent_id)
        self.events_logged += 1

    async def on_tool_called(self, event_data: Dict[str, Any]) -> None:
        """Handle tool called event."""
        tool_name = event_data.get("tool_name", "Unknown")
        args = event_data.get("args", {})
        self.logger.info("Tool called: %s with args: %s", tool_name, args)
        self.events_logged += 1

    async def on_tool_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle tool completed event."""
        tool_name = event_data.get("tool_name", "Unknown")
        duration = event_data.get("duration", 0)
        success = event_data.get("success", False)
        status = "succeeded" if success else "failed"
        self.logger.info("Tool %s %s in %.2fs", tool_name, status, duration)
        self.events_logged += 1

    async def on_llm_request(self, event_data: Dict[str, Any]) -> None:
        """Handle LLM request event."""
        model = event_data.get("model", "Unknown")
        self.logger.info("LLM request sent to model: %s", model)
        self.events_logged += 1

    async def on_llm_response(self, event_data: Dict[str, Any]) -> None:
        """Handle LLM response event."""
        model = event_data.get("model", "Unknown")
        tokens = event_data.get("tokens", 0)
        self.logger.info(
            "LLM response received from model: %s (%s tokens)", model, tokens
        )
        self.events_logged += 1

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated."""
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            "Shutting down %s. Logged %d events in %.2fs",
            self.name,
            self.events_logged,
            elapsed_time,
        )


class MetricsPlugin(Plugin):
    """A plugin that collects and reports metrics on agent performance."""

    def __init__(self) -> None:
        super().__init__(
            name="metrics_plugin",
            version="1.0.0",
            description="Collects and reports metrics on agent performance",
        )
        self.logger = logging.getLogger("fastadk.plugins.metrics")
        self.tool_timings: Dict[str, List[float]] = {}
        self.llm_tokens: Dict[str, int] = {}
        self.start_time = time.time()

    async def initialize(self, plugin_manager: DemoPluginManager) -> None:
        """Initialize the plugin."""
        self.logger.info("Initializing %s v%s", self.name, self.version)
        self.register_event_handlers(plugin_manager)

    def register_event_handlers(self, plugin_manager: DemoPluginManager) -> None:
        """Register event handlers for this plugin."""
        plugin_manager.register_event_handler("tool:called", self.on_tool_called)
        plugin_manager.register_event_handler("tool:completed", self.on_tool_completed)
        plugin_manager.register_event_handler("llm:response", self.on_llm_response)

    async def on_tool_called(self, event_data: Dict[str, Any]) -> None:
        """Record tool start time."""
        # Store the start time in event_data for later use
        event_data["_metrics_start_time"] = time.time()

    async def on_tool_completed(self, event_data: Dict[str, Any]) -> None:
        """Record tool timing."""
        tool_name = event_data.get("tool_name", "unknown")
        start_time = event_data.get("_metrics_start_time")

        if start_time:
            duration = time.time() - start_time

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


class ToolEnhancementPlugin(Plugin):
    """A plugin that enhances agent tools with additional functionality."""

    def __init__(self) -> None:
        super().__init__(
            name="tool_enhancement_plugin",
            version="1.0.0",
            description="Enhances agent tools with additional functionality",
        )
        self.logger = logging.getLogger("fastadk.plugins.tool_enhancement")
        self.enhanced_agents: List[BaseAgent] = []

    async def initialize(self, plugin_manager: DemoPluginManager) -> None:
        """Initialize the plugin."""
        self.logger.info("Initializing %s v%s", self.name, self.version)
        plugin_manager.register_event_handler(
            "agent:initialized", self.on_agent_initialized
        )

    async def on_agent_initialized(self, event_data: Dict[str, Any]) -> None:
        """Enhance agent tools when an agent is initialized."""
        agent = event_data.get("agent")
        if (
            agent
            and hasattr(agent, "register_tool")
            and agent not in self.enhanced_agents
        ):
            self.enhance_agent(agent)
            self.enhanced_agents.append(agent)

    def enhance_agent(self, agent: BaseAgent) -> None:
        """Add additional tools to the agent."""
        self.logger.info("Enhancing agent with additional tools")

        # Define a new tool function
        async def enhanced_search(query: str, limit: int = 5) -> Dict[str, Any]:
            """
            An enhanced search tool added by the plugin.

            Args:
                query: Search query
                limit: Maximum number of results

            Returns:
                Search results
            """
            self.logger.info("Enhanced search called with query: %s", query)

            # Simulate search results
            results = [
                {
                    "title": f"Result {i} for '{query}'",
                    "description": f"This is a simulated search result {i} for query: {query}",
                    "relevance": round(1.0 - (i * 0.15), 2),
                }
                for i in range(1, min(limit + 1, 8))
            ]

            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "source": "enhanced_search_plugin",
                "timestamp": datetime.now().isoformat(),
            }

        # Register the tool with the agent
        agent.register_tool("enhanced_search", enhanced_search)

        # Add a tool usage tracking method
        if not hasattr(agent, "get_tool_usage"):
            agent.tool_usage_counts = {}

            def track_tool_usage(tool_name: str) -> None:
                """Track tool usage."""
                if tool_name not in agent.tool_usage_counts:
                    agent.tool_usage_counts[tool_name] = 0
                agent.tool_usage_counts[tool_name] += 1

            def get_tool_usage() -> Dict[str, int]:
                """Get tool usage statistics."""
                return dict(agent.tool_usage_counts)

            # Add the methods to the agent
            agent.track_tool_usage = track_tool_usage
            agent.get_tool_usage = get_tool_usage

            # Hook into the execute_tool method to track usage
            original_execute_tool = agent.execute_tool

            async def execute_tool_with_tracking(*args, **kwargs):
                """Wrapper to track tool usage."""
                if args and isinstance(args[0], str):
                    agent.track_tool_usage(args[0])
                return await original_execute_tool(*args, **kwargs)

            agent.execute_tool = execute_tool_with_tracking

        self.logger.info("Agent enhancement complete")

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated."""
        self.logger.info("Shutting down %s", self.name)
        self.enhanced_agents = []


@Agent(
    model="gemini-1.5-pro",
    description="An agent that demonstrates the plugin system",
    provider="gemini",  # Will fall back to simulated if no API key is available
    system_prompt="""
    You are a helpful assistant that demonstrates how plugins can extend agent functionality.
    """,
)
class PluginDemoAgent(BaseAgent):
    """Agent demonstrating the plugin system."""

    def __init__(self) -> None:
        super().__init__()
        # In a real application, the plugin manager would be initialized by the framework
        # For this demo, we'll create it manually
        self.plugin_manager = DemoPluginManager()
        self._setup_demo_data()

    def _setup_demo_data(self) -> None:
        """Set up demo data for tools."""
        # Demo weather data
        self.weather_data = {
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
            "houston": {
                "temperature": 92,
                "conditions": "Hot and humid",
                "humidity": 85,
            },
            "miami": {
                "temperature": 88,
                "conditions": "Scattered thunderstorms",
                "humidity": 80,
            },
        }

    @tool(return_type=dict)
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a location.

        Args:
            location: The city to get weather for

        Returns:
            Weather information
        """
        location = location.lower()

        if location in self.weather_data:
            return {
                "location": location,
                "temperature": self.weather_data[location]["temperature"],
                "conditions": self.weather_data[location]["conditions"],
                "humidity": self.weather_data[location]["humidity"],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "error": f"Weather data not available for {location}",
                "available_locations": list(self.weather_data.keys()),
            }

    @tool(return_type=dict)
    def get_time(self, timezone: str = "local") -> Dict[str, Any]:
        """
        Get the current time.

        Args:
            timezone: The timezone to get time for

        Returns:
            Current time information
        """
        current_time = datetime.now()

        return {
            "timestamp": current_time.isoformat(),
            "formatted_time": current_time.strftime("%I:%M:%S %p"),
            "formatted_date": current_time.strftime("%Y-%m-%d"),
            "timezone": timezone,
        }

    @tool(return_type=dict)
    def list_active_plugins(self) -> Dict[str, Any]:
        """
        List all active plugins in the system.

        Returns:
            List of active plugins
        """
        if not hasattr(self, "plugin_manager"):
            return {
                "success": False,
                "message": "Plugin manager not available",
                "plugins": [],
            }

        plugins = []
        for plugin in self.plugin_manager.plugins.values():
            plugins.append(
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                }
            )

        return {
            "success": True,
            "plugin_count": len(plugins),
            "plugins": plugins,
        }

    @tool(return_type=dict)
    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get a metrics report from the metrics plugin.

        Returns:
            Metrics report
        """
        if not hasattr(self, "plugin_manager"):
            return {
                "success": False,
                "message": "Plugin manager not available",
                "metrics": {},
            }

        metrics_plugin = self.plugin_manager.get_plugin("metrics_plugin")

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

    @tool(return_type=dict)
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        Get tool usage statistics added by the enhancement plugin.

        Returns:
            Tool usage statistics
        """
        if hasattr(self, "tool_usage_counts"):
            return {
                "success": True,
                "tool_usage": dict(self.tool_usage_counts),
            }
        else:
            # Provide sample data for demo purposes
            sample_usage = {
                "get_weather": 10,
                "get_time": 5,
                "list_active_plugins": 3,
                "enhanced_search": 2,
            }
            return {
                "success": True,
                "message": "Using sample data (no tracking available)",
                "tool_usage": sample_usage,
            }


async def demonstrate_plugin_system() -> None:
    """Run the plugin system demonstration."""
    print("\n" + "=" * 60)
    print("🔌 FastADK Plugin System Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    try:
        # Create the agent, which will automatically initialize the plugin manager
        print("\n🚀 Initializing agent with plugins...")
        agent = PluginDemoAgent()

        # Create and register plugins
        logging_plugin = LoggingPlugin()
        metrics_plugin = MetricsPlugin()
        tool_enhancement_plugin = ToolEnhancementPlugin()

        print("\n📦 Registering plugins...")
        await agent.plugin_manager.register_plugin(logging_plugin)
        await agent.plugin_manager.register_plugin(metrics_plugin)
        await agent.plugin_manager.register_plugin(tool_enhancement_plugin)

        # List active plugins
        print("\n📋 Active Plugins:")
        result = await agent.execute_tool("list_active_plugins")

        if result.get("success", False):
            plugins = result.get("plugins", [])
            for plugin in plugins:
                print(
                    f"  - {plugin['name']} v{plugin['version']}: {plugin['description']}"
                )
        else:
            print(f"  Error: {result.get('message')}")

        # Demonstrate basic tools (these will trigger plugin events)
        print("\n🧰 Demonstrating Basic Tools:")

        # Get weather
        print("\n🌤️  Weather Tool:")
        locations = ["New York", "Miami", "Unknown City"]

        for location in locations:
            print(f"\n  Getting weather for {location}...")
            result = await agent.execute_tool("get_weather", location=location)

            if "error" not in result:
                print(
                    f"  📍 {location.title()}: {result['temperature']}°F, {result['conditions']}"
                )
                print(f"  🌡️  Humidity: {result['humidity']}%")
            else:
                print(f"  ❌ {result['error']}")
                print(
                    f"  Available locations: {', '.join(result['available_locations'])}"
                )

        # Get time
        print("\n⏰ Time Tool:")
        result = await agent.execute_tool("get_time")
        print(f"  🕒 Current time: {result['formatted_time']}")
        print(f"  📅 Current date: {result['formatted_date']}")

        # Demonstrate plugin-enhanced functionality
        print("\n🔍 Enhanced Search Tool (added by plugin):")
        try:
            result = await agent.execute_tool(
                "enhanced_search", query="artificial intelligence", limit=3
            )

            if "results" in result:
                print(f"  Query: {result['query']}")
                print(f"  Found {result['result_count']} results:")

                for i, item in enumerate(result["results"], 1):
                    print(f"  {i}. {item['title']} (relevance: {item['relevance']})")
                    print(f"     {item['description']}")
            else:
                print("  ❌ Search failed")
        except Exception as e:
            print(f"  ❌ Error using enhanced search: {e}")

        # Get tool usage statistics
        print("\n📊 Tool Usage Statistics (added by plugin):")
        result = await agent.execute_tool("get_tool_usage_stats")

        if result.get("success", False):
            usage = result.get("tool_usage", {})
            for tool_name, count in usage.items():
                print(f"  - {tool_name}: {count} calls")
        else:
            print(f"  ❌ {result.get('message')}")

        # Get metrics report
        print("\n📈 Metrics Report from Plugin:")
        # Make another tool call to generate more metrics
        await agent.execute_tool("get_weather", location="Chicago")

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

        # Clean up and shutdown plugins
        print("\n🧹 Shutting down plugins...")
        await agent.plugin_manager.shutdown()

        print("\n" + "=" * 60)
        print("🏁 FastADK - Plugin System Demo Completed")
        print("=" * 60)
    except Exception as e:
        logger.error("Error in plugin demo: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")


async def main() -> None:
    """Run the main demo."""
    await demonstrate_plugin_system()


if __name__ == "__main__":
    asyncio.run(main())
