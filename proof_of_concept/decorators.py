"""
Proof of concept implementation of FastADK decorators.

This module demonstrates the feasibility of using decorators to simplify
agent and tool definitions in Google ADK. It's a simple implementation
intended to validate the approach, not for production use.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints


def tool(
    _func=None,
    *,
    name: str | None = None,
    description: str | None = None,
    cache_ttl: int = 0,
    timeout: int | None = None,
    retries: int = 0,
):
    """
    Decorator to register a function as a tool for agent use.

    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional description (defaults to function docstring)
        cache_ttl: Time in seconds to cache results (0 means no caching)
        timeout: Maximum execution time in seconds
        retries: Number of retry attempts on failure

    Returns:
        The decorated function with tool metadata attached
    """

    def decorator(func: Callable) -> Callable:
        # Extract tool metadata from function signature and docstring
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Parse parameter descriptions from docstring (simplified)
        param_descriptions = {}
        if func.__doc__:
            for line in func.__doc__.split("\n"):
                line = line.strip()
                if line.startswith(":param ") or line.startswith("@param "):
                    parts = line.split(":", 2) if ":" in line else line.split(" ", 2)
                    if len(parts) >= 3:
                        param_name = parts[1].strip()
                        param_descriptions[param_name] = parts[2].strip()

        # Build parameter schema
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            param_type = (
                type_hints.get(param_name, Any).__name__
                if param_name in type_hints
                else "any"
            )

            parameters[param_name] = {
                "type": param_type,
                "description": param_descriptions.get(param_name, ""),
                "required": param.default is inspect.Parameter.empty,
            }

        # Build return type schema
        return_type = "any"
        if "return" in type_hints:
            return_type = type_hints["return"].__name__

        # Store metadata on the function object for later registration
        func._tool_metadata = {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters,
            "return_type": return_type,
            "cache_ttl": cache_ttl,
            "timeout": timeout,
            "retries": retries,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Here we would add pre/post processing, validation, logging, etc.
            # For now, just call the function directly
            return func(*args, **kwargs)

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


class ProviderABC:
    """Abstract base class for provider backends."""

    def initialize_agent(self, metadata: dict[str, Any]) -> Any:
        """Initialize agent with the provider backend."""
        raise NotImplementedError("Subclasses must implement initialize_agent")

    def register_tool(self, agent: Any, tool_metadata: dict[str, Any]) -> None:
        """Register a tool with the agent."""
        raise NotImplementedError("Subclasses must implement register_tool")

    async def run(self, agent: Any, input_text: str, **kwargs) -> str:
        """Run the agent with the given input."""
        raise NotImplementedError("Subclasses must implement run")


class SimulatedProvider(ProviderABC):
    """Simulated provider for demonstration purposes."""

    def initialize_agent(self, metadata: dict[str, Any]) -> Any:
        """Initialize a simulated agent."""
        return {"type": "simulated_agent", "metadata": metadata, "tools": []}

    def register_tool(
        self, agent: dict[str, Any], tool_metadata: dict[str, Any]
    ) -> None:
        """Register a tool with the simulated agent."""
        agent["tools"].append(tool_metadata)

    async def run(self, agent: dict[str, Any], input_text: str, **kwargs) -> str:
        """Run the simulated agent."""
        # This is a very simplistic simulation that looks for tool names in the input
        # and pretends to execute them
        response = f"Processing: '{input_text}'\n"

        tools_used = []
        for tool in agent["tools"]:
            tool_name = tool["name"].lower()
            # Check if weather-related keywords are in the input for weather tool
            if "get_weather" in tool_name and any(
                kw in input_text.lower()
                for kw in ["weather", "temperature", "forecast"]
            ):
                tools_used.append(tool_name)
                response += f"Using tool: {tool['name']}\n"
                response += "Weather result: 22°C, Sunny\n"
            # Check if search-related keywords are in the input for search tool
            elif "search" in tool_name and any(
                kw in input_text.lower()
                for kw in ["search", "find", "information", "patterns"]
            ):
                tools_used.append(tool_name)
                response += f"Using tool: {tool['name']}\n"
                response += "Search result: Found relevant information\n"

        if tools_used:
            response += f"\nAgent response: Based on the information I found using {', '.join(tools_used)}, "

            if "get_weather" in tools_used:
                response += "the weather is currently 22°C and sunny. "
            if "search_weather_info" in tools_used:
                response += (
                    "I found some weather patterns information that might be helpful. "
                )

            response += (
                f"Is there anything else you'd like to know about '{input_text}'?"
            )
        else:
            response += f"\nAgent response: I've processed your request about '{input_text}', but I don't have specific tools to help with that."

        return response


def Agent(*, model: str, description: str, provider: str = "simulated", **kwargs):
    """
    Class decorator to register a class as an agent.

    Args:
        model: The LLM model to use
        description: Description of the agent's purpose
        provider: Backend provider to use ('adk', 'langchain', 'simulated')
        **kwargs: Additional provider-specific parameters

    Returns:
        The decorated class with agent functionality
    """

    def decorator(cls: type) -> type:
        # Store agent metadata on the class
        cls._agent_metadata = {
            "model": model,
            "description": description,
            "provider_name": provider,
            **kwargs,
        }

        # Find and register all methods decorated with @tool
        cls._tools = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, "_tool_metadata"):
                cls._tools[attr_name] = attr._tool_metadata

        # Enhance __init__ to register with provider
        original_init = cls.__init__

        @functools.wraps(original_init)
        def init_wrapper(self, *args, **kwargs):
            # Select provider based on name
            if provider == "simulated":
                self.provider = SimulatedProvider()
            else:
                # In a real implementation, we would have a provider factory
                # that would return the appropriate provider
                self.provider = SimulatedProvider()

            # Initialize with original __init__
            original_init(self, *args, **kwargs)

            # Initialize agent with provider
            self.agent = self.provider.initialize_agent(cls._agent_metadata)

            # Register tools
            for tool_metadata in cls._tools.values():
                self.provider.register_tool(self.agent, tool_metadata)

        # Add run method if it doesn't exist
        if not hasattr(cls, "run") or not callable(cls.run):

            async def run(self, input_text: str, **kwargs) -> str:
                """Run the agent with the given input."""
                return await self.provider.run(self.agent, input_text, **kwargs)

            cls.run = run

        cls.__init__ = init_wrapper
        return cls

    return decorator


# Example usage of the decorators
if __name__ == "__main__":

    @Agent(model="gemini-2.0", description="Weather assistant")
    class WeatherAgent:
        @tool
        def get_weather(self, city: str) -> dict:
            """
            Fetch current weather for a city.

            :param city: The name of the city to get weather for
            :return: Weather information
            """
            # In a real implementation, this would call a weather API
            return {"city": city, "temp": "22°C", "condition": "sunny"}

        @tool(name="search_weather_info", cache_ttl=300)
        def search(self, query: str) -> str:
            """
            Search for weather information online.

            :param query: The search query
            :return: Search results
            """
            return f"Results for {query}"

    import asyncio

    async def test_agent():
        agent = WeatherAgent()
        print("Agent metadata:", agent.agent["metadata"])
        print("Registered tools:")
        for tool in agent.agent["tools"]:
            print(f"  - {tool['name']}: {tool['description']}")

        response = await agent.run("What's the weather in Paris?")
        print("\nResponse:")
        print(response)

        response = await agent.run("Can you search for weather patterns in Europe?")
        print("\nResponse:")
        print(response)

    asyncio.run(test_agent())
