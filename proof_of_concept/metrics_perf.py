"""
Benchmark for FastADK proof of concept.

This module compares the performance and developer experience of using
the FastADK decorators versus a simulated "raw" approach.
"""

import asyncio
import inspect
import statistics
import time
import tracemalloc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# Import our decorator-based implementation
from proof_of_concept.decorators import Agent, tool

# =====================================================================
# Utility functions used by both implementations
# =====================================================================


def get_weather_func(city: str) -> dict[str, str]:
    """Get weather for a city."""
    return {"city": city, "temp": "22°C", "condition": "sunny"}


def search_func(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"


# =====================================================================
# Simulated "raw" implementation (what developers might write without FastADK)
# =====================================================================


@dataclass
class RawTool:
    """A tool in the raw implementation."""

    name: str
    description: str
    func: Callable
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


class RawAgent:
    """A simulated raw agent without decorator abstractions."""

    def __init__(self, model: str, description: str):
        self.model = model
        self.description = description
        self.tools = []
        self.agent_data = {
            "type": "simulated_agent",
            "metadata": {"model": model, "description": description},
            "tools": [],
        }

    def add_tool(self, tool: RawTool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        self.agent_data["tools"].append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        )

    async def run(self, input_text: str) -> str:
        """Run the agent."""
        response = f"Processing: '{input_text}'\n"

        tools_used = []
        for tool_item in self.tools:
            tool_name = tool_item.name.lower()
            if "get_weather" in tool_name and any(
                kw in input_text.lower()
                for kw in ["weather", "temperature", "forecast"]
            ):
                tools_used.append(tool_name)
                response += f"Using tool: {tool.name}\n"
                result = tool.func("Paris")  # Simplified for benchmark
                response += f"Weather result: {result['temp']}, {result['condition']}\n"
            elif "search" in tool_name and any(
                kw in input_text.lower()
                for kw in ["search", "find", "information", "patterns"]
            ):
                tools_used.append(tool_name)
                response += f"Using tool: {tool.name}\n"
                result = tool.func("weather patterns")  # Simplified for benchmark
                response += f"Search result: {result}\n"

        if tools_used:
            response += f"\nAgent response: Based on the information I found using {', '.join(tools_used)}, "
            if any("get_weather" in t for t in tools_used):
                response += "the weather is currently 22°C and sunny. "
            if any("search" in t for t in tools_used):
                response += (
                    "I found some weather patterns information that might be helpful. "
                )
            response += (
                f"Is there anything else you'd like to know about '{input_text}'?"
            )
        else:
            response += f"\nAgent response: I've processed your request about '{input_text}', but I don't have specific tools to help with that."

        return response


# =====================================================================
# Setup functions for both implementations
# =====================================================================


def setup_raw_agent():
    """Set up a raw agent."""
    agent = RawAgent(model="gemini-2.0", description="Weather assistant")

    # Create and add weather tool
    weather_tool = RawTool(
        name="get_weather",
        description="Fetch current weather for a city",
        func=get_weather_func,
        parameters={
            "city": {
                "type": "str",
                "description": "The name of the city to get weather for",
                "required": True,
            }
        },
    )
    agent.add_tool(weather_tool)

    # Create and add search tool
    search_tool = RawTool(
        name="search_weather_info",
        description="Search for weather information online",
        func=search_func,
        parameters={
            "query": {
                "type": "str",
                "description": "The search query",
                "required": True,
            }
        },
    )
    agent.add_tool(search_tool)

    return agent


def setup_decorator_agent():
    """Set up an agent using decorators."""

    @Agent(model="gemini-2.0", description="Weather assistant")
    class WeatherAgent:
        @tool
        def get_weather(self, city: str) -> dict[str, str]:
            """
            Fetch current weather for a city.

            :param city: The name of the city to get weather for
            :return: Weather information
            """
            return get_weather_func(city)

        @tool(name="search_weather_info")
        def search(self, query: str) -> str:
            """
            Search for weather information online.

            :param query: The search query
            :return: Search results
            """
            return search_func(query)

    return WeatherAgent()


# =====================================================================
# Benchmark functions
# =====================================================================


def measure_setup_time(setup_func: Callable, iterations: int = 10) -> dict[str, float]:
    """
    Measure the time it takes to set up an agent over multiple iterations.

    Returns dict with: mean, median, min, max, stdev
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()
        setup_func()
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def measure_memory_usage(func: Callable, iterations: int = 5) -> dict[str, Any]:
    """
    Measure the memory usage of a function over multiple iterations.

    Returns dict with statistics and the last result object.
    """
    peaks = []
    result = None

    for _ in range(iterations):
        tracemalloc.start()
        result = func()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

    return {
        "mean": statistics.mean(peaks),
        "median": statistics.median(peaks),
        "min": min(peaks),
        "max": max(peaks),
        "stdev": statistics.stdev(peaks) if len(peaks) > 1 else 0,
        "result": result,
    }


async def measure_execution_time(
    func: Callable[..., Awaitable[str]], input_text: str, iterations: int = 20
) -> dict[str, float]:
    """
    Measure the execution time of an async function over multiple iterations.

    Returns dict with: mean, median, min, max, stdev
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()
        await func(input_text)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


# =====================================================================
# Main benchmark function
# =====================================================================


def print_stats(name: str, stats: dict[str, float], unit: str = "seconds"):
    """Helper function to print statistics in a consistent format."""
    print(f"{name}:")
    print(f"  Mean: {stats['mean']:.6f} {unit}")
    print(f"  Median: {stats['median']:.6f} {unit}")
    print(f"  Min: {stats['min']:.6f} {unit}")
    print(f"  Max: {stats['max']:.6f} {unit}")
    print(f"  StdDev: {stats['stdev']:.6f} {unit}")


async def run_benchmarks():
    """Run benchmarks comparing the two implementations."""
    print("\n==== FastADK Proof of Concept Benchmark ====\n")
    print(
        "Running extended benchmarks with multiple iterations for statistical significance\n"
    )

    # Setup time
    print("Measuring setup time...")
    raw_setup_stats = measure_setup_time(setup_raw_agent, iterations=50)
    decorator_setup_stats = measure_setup_time(setup_decorator_agent, iterations=50)

    print("\nRaw implementation setup time")
    print_stats("Setup time", raw_setup_stats)

    print("\nDecorator implementation setup time")
    print_stats("Setup time", decorator_setup_stats)

    setup_diff_pct = (
        (raw_setup_stats["mean"] - decorator_setup_stats["mean"])
        / raw_setup_stats["mean"]
        * 100
    )
    print(f"\nSetup time difference (mean): {setup_diff_pct:.2f}%")

    # Memory usage
    print("\nMeasuring memory usage...")
    raw_memory_stats = measure_memory_usage(setup_raw_agent, iterations=20)
    decorator_memory_stats = measure_memory_usage(setup_decorator_agent, iterations=20)

    raw_agent = raw_memory_stats["result"]
    decorator_agent = decorator_memory_stats["result"]

    print("\nRaw implementation memory usage")
    print_stats(
        "Memory",
        {k: v / 1024 for k, v in raw_memory_stats.items() if k != "result"},
        "KB",
    )

    print("\nDecorator implementation memory usage")
    print_stats(
        "Memory",
        {k: v / 1024 for k, v in decorator_memory_stats.items() if k != "result"},
        "KB",
    )

    memory_diff_pct = (
        (decorator_memory_stats["mean"] - raw_memory_stats["mean"])
        / raw_memory_stats["mean"]
        * 100
    )
    print(f"\nMemory usage difference (mean): {memory_diff_pct:.2f}%")

    # Execution time
    print("\nMeasuring execution time...")

    test_inputs = [
        "What's the weather in Paris?",
        "Can you search for weather patterns in Europe?",
        "Tell me about both the weather and search for climate data",
    ]

    print("\nExecution time by query:")

    raw_time_stats_by_query = []
    decorator_time_stats_by_query = []

    for i, input_text in enumerate(test_inputs):
        print(f"\nQuery {i + 1}: '{input_text}'")

        # Raw implementation
        raw_time_stats = await measure_execution_time(
            raw_agent.run, input_text, iterations=30
        )
        raw_time_stats_by_query.append(raw_time_stats)
        print("\nRaw implementation")
        print_stats("Execution time", raw_time_stats)

        # Decorator implementation
        decorator_time_stats = await measure_execution_time(
            decorator_agent.run, input_text, iterations=30
        )
        decorator_time_stats_by_query.append(decorator_time_stats)
        print("\nDecorator implementation")
        print_stats("Execution time", decorator_time_stats)

        exec_diff_pct = (
            (decorator_time_stats["mean"] - raw_time_stats["mean"])
            / raw_time_stats["mean"]
            * 100
        )
        print(f"\nExecution time difference (mean): {exec_diff_pct:.2f}%")

    # Calculate overall execution time statistics
    raw_means = [stats["mean"] for stats in raw_time_stats_by_query]
    decorator_means = [stats["mean"] for stats in decorator_time_stats_by_query]

    print("\nOverall execution time comparison:")
    print(f"Raw implementation mean: {statistics.mean(raw_means):.6f} seconds")
    print(
        f"Decorator implementation mean: {statistics.mean(decorator_means):.6f} seconds"
    )
    overall_diff_pct = (
        (statistics.mean(decorator_means) - statistics.mean(raw_means))
        / statistics.mean(raw_means)
        * 100
    )
    print(f"Overall execution time difference: {overall_diff_pct:.2f}%")

    # Code complexity comparison
    raw_loc = len(inspect.getsource(RawAgent).split("\n"))
    raw_loc += len(inspect.getsource(RawTool).split("\n"))
    raw_loc += len(inspect.getsource(setup_raw_agent).split("\n"))

    decorator_loc = len(inspect.getsource(setup_decorator_agent).split("\n"))

    print("\nCode complexity comparison:")
    print(f"Raw implementation lines of code: {raw_loc}")
    print(f"Decorator implementation lines of code: {decorator_loc}")
    code_reduction_pct = (raw_loc - decorator_loc) / raw_loc * 100
    print(f"Code reduction: {code_reduction_pct:.2f}%")

    print("\n==== Benchmark Complete ====\n")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
