"""
Example demonstrating the use of the fine-tuning utilities in FastADK.

This example shows how to:
1. Convert data between different formats
2. Create and monitor fine-tuning jobs
3. Use the resulting fine-tuned models

Requirements:
- An OpenAI API key with access to fine-tuning endpoints

Note: Running this example will create real fine-tuning jobs that may incur costs.
The example is set up to use minimal data and a small model to keep costs low.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastadk.training import (
    DataConverter,
    DataFormat,
    FineTuningConfig,
    FineTuningProvider,
    default_fine_tuner,
)

# Sample data in Alpaca format for conversion
SAMPLE_DATA = [
    {
        "instruction": "Summarize the following text.",
        "input": "FastADK is a developer-friendly framework for building AI agents. It provides high-level abstractions, declarative APIs, and developer-friendly tooling.",
        "output": "FastADK is a framework that simplifies AI agent development with high-level abstractions and developer-friendly APIs.",
    },
    {
        "instruction": "Write a haiku about programming.",
        "input": "",
        "output": "Fingers on keyboard\nLogic flows through lines of code\nBugs hide in shadows",
    },
]


async def example_data_conversion():
    """Example demonstrating data format conversion."""
    print("\n=== Data Conversion Example ===")

    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample data files
        alpaca_file = temp_path / "alpaca_data.json"
        openai_file = temp_path / "openai_data.jsonl"
        vertex_file = temp_path / "vertex_data.json"

        # Write sample Alpaca data
        with open(alpaca_file, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_DATA, f, indent=2)

        print(f"Created sample Alpaca data file: {alpaca_file}")

        # Convert from Alpaca to OpenAI format
        print("\nConverting from Alpaca to OpenAI format...")
        DataConverter.convert(
            input_file=alpaca_file,
            output_file=openai_file,
            input_format=DataFormat.ALPACA,
            output_format=DataFormat.OPENAI,
        )

        # Display the converted data
        print("\nConverted OpenAI format data:")
        with open(openai_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                print(f"- Message exchange with {len(data['messages'])} messages")

        # Convert from Alpaca to Vertex format
        print("\nConverting from Alpaca to Vertex format...")
        DataConverter.convert(
            input_file=alpaca_file,
            output_file=vertex_file,
            input_format=DataFormat.ALPACA,
            output_format=DataFormat.VERTEX,
        )

        # Display the converted data
        print("\nConverted Vertex format data:")
        with open(vertex_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                print(
                    f"- Example: '{item['input_text'][:50]}...' -> '{item['output_text'][:50]}...'"
                )


async def example_fine_tuning_job(run_actual_job=False):
    """
    Example demonstrating fine-tuning job creation and monitoring.

    Args:
        run_actual_job: Set to True to actually create a fine-tuning job
                        (requires API key and will incur costs)
    """
    print("\n=== Fine-Tuning Job Example ===")

    # Check which providers are available
    providers = default_fine_tuner.supported_providers()
    print(f"Available fine-tuning providers: {[p.value for p in providers]}")

    if FineTuningProvider.OPENAI in providers and run_actual_job:
        # Check if we have an OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Skipping actual job creation: OPENAI_API_KEY environment variable not set"
            )
            return

        # Create a temporary directory for our files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample data files
            alpaca_file = temp_path / "alpaca_data.json"
            openai_file = temp_path / "openai_data.jsonl"

            # Write sample Alpaca data
            with open(alpaca_file, "w", encoding="utf-8") as f:
                json.dump(SAMPLE_DATA, f, indent=2)

            # Convert to OpenAI format
            DataConverter.convert(
                input_file=alpaca_file,
                output_file=openai_file,
                input_format=DataFormat.ALPACA,
                output_format=DataFormat.OPENAI,
            )

            # Configure fine-tuning job
            config = FineTuningConfig(
                provider=FineTuningProvider.OPENAI,
                base_model="gpt-3.5-turbo",  # Use the smallest model to minimize costs
                training_file=str(openai_file),
                hyperparameters={"n_epochs": 1},  # Minimum training to minimize costs
            )

            print(f"Creating fine-tuning job with model: {config.base_model}")

            # Create the job
            job = await default_fine_tuner.create_job(config)
            print(f"Created job with ID: {job.job_id}")
            print(f"Job status: {job.status}")
            print("Check the OpenAI dashboard for job progress")

            # In a real application, you would poll for job status
            print("\nTo check job status later, use:")
            print(
                f"job = await default_fine_tuner.get_job('{job.job_id}', FineTuningProvider.OPENAI)"
            )
            print("print(f'Job status: {job.status}')")
    else:
        print("Skipping actual job creation (run_actual_job=False)")
        print("\nTo create an actual fine-tuning job:")
        print("1. Install the required packages: uv add openai")
        print("2. Set your API key: export OPENAI_API_KEY=your_key_here")
        print("3. Call this function with run_actual_job=True")


async def main():
    """Run the examples."""
    # Example 1: Data conversion
    await example_data_conversion()

    # Example 2: Fine-tuning job (disabled by default to avoid costs)
    await example_fine_tuning_job(run_actual_job=False)


if __name__ == "__main__":
    asyncio.run(main())
