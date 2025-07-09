"""
FastADK Workflow Orchestration Demo.

This example demonstrates the use of workflow orchestration features in FastADK.
It shows how to:
1. Create sequential and parallel workflows
2. Use step decorators to define workflow steps
3. Create conditional branches in workflows
4. Handle errors and retries
5. Compose multiple workflow steps together

To run this example:
```
uv run workflow_demo.py
```

No external API keys or dependencies are required for this example
as it runs entirely with simulated data.
"""

import asyncio
import random
import time
from typing import Any

from fastadk.core.workflow import (
    ParallelFlow,
    SequentialFlow,
    Workflow,
    conditional,
    merge,
    step,
    transform,
)


@step(name="Data Loader", timeout=5)
async def load_data(data_source: str) -> dict[str, Any]:
    """
    Load data from a source.

    Args:
        data_source: The source of the data

    Returns:
        Loaded data
    """
    print(f"🔄 Loading data from {data_source}...")
    await asyncio.sleep(0.5)  # Simulate data loading

    # Return different sample data based on source
    if data_source == "weather":
        return {
            "location": "New York",
            "temperature": 72,
            "conditions": "sunny",
            "forecast": [
                {"day": "Monday", "temp": 75, "conditions": "sunny"},
                {"day": "Tuesday", "temp": 70, "conditions": "cloudy"},
                {"day": "Wednesday", "temp": 65, "conditions": "rainy"},
            ],
        }
    elif data_source == "finance":
        return {
            "symbol": "AAPL",
            "price": 178.72,
            "change": 1.25,
            "volume": 34500000,
            "history": [
                {"date": "2025-07-01", "price": 177.47},
                {"date": "2025-07-02", "price": 178.72},
                {"date": "2025-07-03", "price": 180.15},
            ],
        }
    else:
        return {"error": "Unknown data source", "source": data_source}


@step(name="Data Validator", retry=2)
async def validate_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate data structure and values.

    Args:
        data: The data to validate

    Returns:
        Validated data with quality score
    """
    print("🔍 Validating data...")

    # Randomly fail sometimes to demonstrate retry
    # For parallel workflow, make retry failure probability lower to ensure successful completion
    failure_probability = 0.2

    # If this is part of a parallel workflow, we can detect it by checking if data is a list
    # This check isn't ideal but works for demonstration purposes
    if isinstance(data, dict) and any(
        key in data for key in ["temperature", "price", "symbol", "location"]
    ):
        # Lower failure probability for key examples
        failure_probability = 0.1

    if random.random() < failure_probability:
        print("⚠️ Validation error (will retry)...")
        raise ValueError("Random validation failure")

    # Check for required fields
    if "error" in data:
        data["valid"] = False
        data["quality_score"] = 0
        return data

    # Add validation metadata
    data["valid"] = True
    data["quality_score"] = random.uniform(0.7, 1.0)

    return data


@transform(name="Data Enricher")
def enrich_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich data with additional information.

    Args:
        data: The data to enrich

    Returns:
        Enriched data
    """
    print("✨ Enriching data...")

    # Skip if data is invalid
    if not data.get("valid", False):
        return data

    # Add timestamp
    data["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Add additional info based on data type
    if "temperature" in data:
        # Weather data
        temp = data["temperature"]
        data["comfort_level"] = (
            "cold" if temp < 60 else "comfortable" if temp < 80 else "hot"
        )
    elif "price" in data:
        # Financial data
        data["market_cap"] = data["price"] * 16_500_000_000  # Approx. AAPL shares

    return data


@step(name="Weather Analyzer")
async def analyze_weather(data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze weather data for insights.

    Args:
        data: Weather data to analyze

    Returns:
        Data with weather analysis
    """
    print("🌤️ Analyzing weather data...")
    await asyncio.sleep(0.3)  # Simulate processing

    if not data.get("valid", False):
        return data

    # Add weather analysis
    temp = data["temperature"]
    conditions = data["conditions"]

    data["analysis"] = {
        "recommendation": (
            "Great day to be outside!"
            if temp > 70 and conditions == "sunny"
            else (
                "Good day for indoor activities."
                if conditions in ["rainy", "snowy"]
                else "Average day, no special recommendations."
            )
        ),
        "temperature_trend": (
            "warming"
            if all(f["temp"] <= data["temperature"] for f in data["forecast"])
            else (
                "cooling"
                if all(f["temp"] >= data["temperature"] for f in data["forecast"])
                else "stable"
            )
        ),
    }

    return data


@step(name="Financial Analyzer")
async def analyze_finance(data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze financial data for insights.

    Args:
        data: Financial data to analyze

    Returns:
        Data with financial analysis
    """
    print("📈 Analyzing financial data...")
    await asyncio.sleep(0.4)  # Simulate processing

    if not data.get("valid", False):
        return data

    # Add financial analysis
    price = data["price"]
    change = data["change"]
    history = data["history"]

    # Calculate price trend percentage
    price_trend = (
        (history[-1]["price"] - history[0]["price"]) / history[0]["price"] * 100
    )

    # Calculate volatility based on price and trend
    volatility_factor = abs(change) / price * 100

    data["analysis"] = {
        "recommendation": (
            "Strong buy"
            if change > 0 and price_trend > 1
            else "Hold" if -0.5 <= change <= 0.5 else "Sell" if change < -1 else "Watch"
        ),
        "price_trend": f"{price_trend:.2f}% over last {len(history)} days",
        "volatility": (
            "high"
            if volatility_factor > 2
            else "medium" if volatility_factor > 1 else "low"
        ),
    }

    return data


@merge(name="Results Formatter")
def format_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Format results from multiple analyzers.

    Args:
        results: List of results from analyzers

    Returns:
        Formatted output
    """
    print("📊 Formatting final results...")

    # Combine all results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": len(results),
        "insights": [],
    }

    for result in results:
        # Debug print to help diagnose issues
        print(
            f"Processing result: valid={result.get('valid')}, has_analysis={'analysis' in result}"
        )

        if result.get("valid", False) and "analysis" in result:
            # Add source-specific insights
            if "temperature" in result:
                source_type = "weather"
                location = result.get("location", "Unknown")
                insight = (
                    f"Weather in {location}: {result['temperature']}°F, {result['conditions']}. "
                    f"{result['analysis'].get('recommendation', '')}"
                )
                output["insights"].append(insight)
                print(f"Added weather insight: {insight}")
            elif "price" in result:
                source_type = "finance"
                symbol = result.get("symbol", "Unknown")
                insight = (
                    f"Stock {symbol}: ${result['price']:.2f} ({result['change']:+.2f}). "
                    f"Recommendation: {result['analysis'].get('recommendation', '')}"
                )
                output["insights"].append(insight)
                print(f"Added finance insight: {insight}")
            else:
                source_type = "unknown"

            # Add the data with source type
            output[f"{source_type}_data"] = {
                k: v
                for k, v in result.items()
                if k not in ("valid", "quality_score", "processed_at")
            }

    # Add summary if we have multiple sources
    if len(output["insights"]) > 1:
        output["summary"] = "Multiple data sources analyzed successfully."
    elif len(output["insights"]) == 1:
        output["summary"] = "One data source analyzed successfully."
    else:
        output["summary"] = "No valid data sources were analyzed."

    return output


async def run_weather_workflow():
    """Run a weather data workflow."""
    print("\n🌟 Running Weather Workflow")
    print("========================")

    # Create a sequential workflow for weather data
    weather_flow = Workflow.sequence(
        load_data,
        validate_data,
        enrich_data,
        analyze_weather,
        name="Weather Analysis Workflow",
    )

    # Execute the workflow
    result = await weather_flow.execute("weather")

    print("\n✅ Weather Workflow Result:")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(
        "Analysis:",
        result.output.get("analysis", {}).get("recommendation", "No analysis"),
    )
    print("========================\n")

    return result.output


async def run_finance_workflow():
    """Run a financial data workflow."""
    print("\n🌟 Running Finance Workflow")
    print("========================")

    # Create workflow with conditional branch
    # This checks the data quality and only analyzes if quality is high enough

    # First define a condition function
    def is_high_quality(data):
        return data.get("quality_score", 0) >= 0.8

    # Create steps with existing functions (not creating new ones)
    # Just creating references to the step objects
    data_load = load_data
    validation = validate_data
    enrichment = enrich_data
    analysis = analyze_finance

    # Create a conditional step
    quality_branch = conditional(
        condition=is_high_quality,
        true_step=analysis,
        false_step=transform(
            lambda d: {**d, "analysis": {"recommendation": "Insufficient data quality"}}
        ),
        name="Quality Check",
    )

    # Create the full workflow
    finance_flow = data_load >> validation >> enrichment >> quality_branch
    workflow = Workflow(root_step=finance_flow, name="Finance Analysis Workflow")

    # Execute the workflow
    result = await workflow.execute("finance")

    print("\n✅ Finance Workflow Result:")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(
        "Analysis:",
        result.output.get("analysis", {}).get("recommendation", "No analysis"),
    )
    print("========================\n")

    return result.output


async def run_parallel_workflow():
    """Run a parallel workflow that processes multiple data sources at once."""
    print("\n🌟 Running Parallel Analysis Workflow")
    print("==================================")

    # Define each branch of our parallel workflow
    weather_branch = load_data >> validate_data >> enrich_data >> analyze_weather

    finance_branch = load_data >> validate_data >> enrich_data >> analyze_finance

    # Create parallel workflow with merge step
    # Create a parallel flow manually
    parallel_flow = ParallelFlow(steps=[weather_branch, finance_branch])
    # Add a merge step after the parallel flow
    flow = SequentialFlow(steps=[parallel_flow, format_results])
    # Create the workflow
    workflow = Workflow(root_step=flow, name="Multi-Source Analysis Workflow")

    # Execute the workflow with different inputs for each branch
    result = await workflow.execute(["weather", "finance"])

    print("\n✅ Parallel Workflow Result:")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Insights: {len(result.output.get('insights', []))} found")
    for insight in result.output.get("insights", []):
        print(f"  - {insight}")
    print("==================================\n")

    return result.output


async def main():
    """Run all workflow demos."""
    print("🚀 FastADK Workflow Orchestration Demo")
    print("=======================================")

    # Run all workflows
    await run_weather_workflow()
    await run_finance_workflow()
    await run_parallel_workflow()

    print("🏁 All workflow demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
