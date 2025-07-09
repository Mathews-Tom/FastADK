"""
Batch Processing Demo for FastADK.

This example demonstrates how to use FastADK's batch processing capabilities to:
1. Process multiple inputs efficiently
2. Handle parallelism with configurable batch sizes
3. Monitor and report progress
4. Apply post-processing to results
5. Handle errors gracefully in batch contexts

Usage:
    1. Set up API keys if you want to use live providers:
        export GEMINI_API_KEY=your_api_key_here

    2. Run the example:
        uv run examples/advanced/batch_processing_demo.py
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool
from fastadk.core.batch import BatchResult, BatchUtils

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@Agent(
    model="gemini-1.5-pro",
    description="A sentiment analysis agent",
    provider="gemini",  # Will fall back to simulated if no API key is available
    system_prompt="""
    You are a sentiment analysis expert. When analyzing text:
    - Consider tone, emotion, and language used
    - Provide a sentiment score on a scale from -1.0 (very negative) to 1.0 (very positive)
    - Include a brief explanation of your reasoning
    - Be objective and consistent in your analysis
    """,
)
class SentimentAnalysisAgent(BaseAgent):
    """
    Agent for analyzing sentiment in text.

    This agent demonstrates batch processing for analyzing the sentiment
    of multiple text inputs efficiently.
    """

    @tool
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        # In a real implementation, this would use the model's understanding,
        # but for demonstration purposes we'll simulate sentiment analysis
        # with some randomness but weighted by positive/negative words

        # Check for common positive and negative terms
        positive_words = [
            "good",
            "great",
            "excellent",
            "happy",
            "love",
            "wonderful",
            "amazing",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "sad",
            "hate",
            "disappointing",
            "horrible",
        ]

        # Count occurrences
        positive_count = sum(word in text.lower() for word in positive_words)
        negative_count = sum(word in text.lower() for word in negative_words)

        # Calculate base sentiment
        if positive_count > negative_count:
            base_sentiment = random.uniform(0.3, 1.0)
            sentiment_label = "positive"
        elif negative_count > positive_count:
            base_sentiment = random.uniform(-1.0, -0.3)
            sentiment_label = "negative"
        else:
            base_sentiment = random.uniform(-0.3, 0.3)
            sentiment_label = "neutral"

        # Add some randomness but stay within bounds
        sentiment_score = max(
            -1.0, min(1.0, base_sentiment + random.uniform(-0.2, 0.2))
        )

        # Return sentiment analysis result
        return {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": sentiment_label,
            "confidence": random.uniform(0.7, 0.95),
            "timestamp": datetime.now().isoformat(),
        }

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a single text input.

        This method wraps the tool execution to provide a consistent interface
        for batch processing.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Simulate processing delay for demonstration purposes
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Execute the sentiment analysis tool
            result = await self.execute_tool("analyze_sentiment", text=text)
            return result
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.error("Error processing text: %s", e)
            return {
                "text": text[:50] + "..." if len(text) > 50 else text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


async def run_batch_processing_demo() -> None:
    """Run the batch processing demonstration."""
    print("\n" + "=" * 60)
    print("📊 FastADK Batch Processing Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    # Initialize the agent
    agent = SentimentAnalysisAgent()

    # Sample data for batch processing
    sample_texts = [
        "I absolutely loved the movie! It was fantastic and exceeded all my expectations.",
        "The customer service was terrible. I waited for hours and still didn't get help.",
        "The product is okay, nothing special but it gets the job done.",
        "I am extremely disappointed with this purchase. Complete waste of money.",
        "Today was a wonderful day! Everything went perfectly and I'm so happy.",
        "The meeting was fine, we covered all the agenda items efficiently.",
        "This restaurant has the best food I've ever tasted! Amazing experience!",
        "I've had better experiences with other companies, this one was just average.",
        "I'm frustrated with how complicated this software is to use.",
        "What a beautiful day! The weather is perfect and everything is going well.",
        "This flight was delayed by three hours and there was no communication.",
        "The concert was alright, some songs were good but overall not very memorable.",
        "I'm really impressed with the quality of this product. Highly recommended!",
        "This book was so boring I couldn't even finish it.",
        "The vacation was nice but the accommodations weren't as advertised.",
    ]

    print(f"\n🔍 Processing {len(sample_texts)} text samples for sentiment analysis...")

    # Simple sequential processing
    print("\n⏱️  Sequential Processing:")
    start_time = time.time()
    sequential_results = []
    for i, text in enumerate(sample_texts, 1):
        print(f"  Processing item {i}/{len(sample_texts)}...")
        result = await agent.process_text(text)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"  ✅ Completed in {sequential_time:.2f} seconds")

    # Batch processing with BatchProcessor
    print("\n⚡ Batch Processing:")

    # Set up for batch processing with parallel processing
    max_concurrency = 5  # Process 5 items in parallel

    # Define a callback for progress updates
    async def progress_callback(completed: int, total: int) -> None:
        """Callback for batch progress updates."""
        percentage = (completed / total) * 100
        print(f"  Progress: {completed}/{total} ({percentage:.1f}%)")

    # Start batch processing
    start_time = time.time()
    # Process with progress updates
    progress_count = 0

    async def process_with_progress(text: str) -> Dict[str, Any]:
        """Wrapper to track progress while processing."""
        nonlocal progress_count
        result = await agent.process_text(text)
        progress_count += 1
        await progress_callback(progress_count, len(sample_texts))
        return result

    batch_result = await BatchUtils.process_parallel(
        items=sample_texts,
        process_func=process_with_progress,
        max_concurrency=max_concurrency,
    )
    # Get results for easier access
    results = [result for _, result in batch_result.successful]
    batch_time = time.time() - start_time

    # Print batch processing results
    print(f"  ✅ Batch completed in {batch_time:.2f} seconds")
    print(f"  📈 Speed improvement: {sequential_time / batch_time:.1f}x faster")
    print(f"  ✓ Successful: {batch_result.success_count}")
    print(f"  ✗ Failed: {batch_result.failure_count}")

    # Analyze the results
    print("\n📊 Sentiment Analysis Results:")

    # Calculate sentiment distribution
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "error": 0}
    for result in results:
        if "error" in result:
            sentiment_counts["error"] += 1
        elif "sentiment_label" in result:
            sentiment_counts[result["sentiment_label"]] += 1

    # Print distribution
    total = len(results)
    print(
        f"  Positive: {sentiment_counts['positive']} ({sentiment_counts['positive'] / total * 100:.1f}%)"
    )
    print(
        f"  Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral'] / total * 100:.1f}%)"
    )
    print(
        f"  Negative: {sentiment_counts['negative']} ({sentiment_counts['negative'] / total * 100:.1f}%)"
    )
    if sentiment_counts["error"] > 0:
        print(
            f"  Errors: {sentiment_counts['error']} ({sentiment_counts['error'] / total * 100:.1f}%)"
        )

    # Demonstrate batch with custom post-processing
    print("\n🔄 Batch Processing with Custom Post-processing:")

    # Define a post-processing function
    def post_process(batch_results: BatchResult) -> Dict[str, Any]:
        """
        Post-process batch results to generate a summary.

        Args:
            batch_results: BatchResult from processing

        Returns:
            Summary of the batch results
        """
        # Extract results from successful operations
        results = [result for _, result in batch_results.successful]
        # Count sentiment categories
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "error": 0}
        # Track average scores
        total_score = 0.0
        valid_count = 0

        # Analyze results
        for result in results:
            if "error" in result:
                sentiment_counts["error"] += 1
            elif "sentiment_label" in result:
                sentiment_counts[result["sentiment_label"]] += 1
                total_score += result.get("sentiment_score", 0)
                valid_count += 1

        # Calculate averages
        avg_score = total_score / valid_count if valid_count > 0 else 0

        # Determine overall sentiment
        if avg_score > 0.2:
            overall_sentiment = "Positive"
        elif avg_score < -0.2:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Create summary
        return {
            "total_items": len(results),
            "valid_items": valid_count,
            "error_count": sentiment_counts["error"],
            "average_sentiment_score": round(avg_score, 2),
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": {
                "positive": sentiment_counts["positive"],
                "neutral": sentiment_counts["neutral"],
                "negative": sentiment_counts["negative"],
            },
            "timestamp": datetime.now().isoformat(),
        }

    # Run batch processing
    start_time = time.time()
    new_batch_result = await BatchUtils.process_parallel(
        items=sample_texts,
        process_func=agent.process_text,
        max_concurrency=max_concurrency,
    )
    # Apply post-processing to get summary
    summary = post_process(new_batch_result)
    batch_time = time.time() - start_time

    # Print summary
    print(f"  ✅ Completed in {batch_time:.2f} seconds")
    print("\n📋 Batch Summary:")
    print(f"  Total items processed: {summary['total_items']}")
    print(f"  Average sentiment score: {summary['average_sentiment_score']}")
    print(f"  Overall sentiment: {summary['overall_sentiment']}")
    print("  Distribution:")
    for sentiment, count in summary["sentiment_distribution"].items():
        percentage = (count / summary["total_items"]) * 100
        print(f"    - {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("🏁 FastADK - Batch Processing Demo Completed")
    print("=" * 60)


async def main() -> None:
    """Run the main demo."""
    await run_batch_processing_demo()


if __name__ == "__main__":
    asyncio.run(main())
