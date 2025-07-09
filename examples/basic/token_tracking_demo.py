"""
Token Tracking Demo for FastADK.

This example demonstrates how to use the token tracking functionality
in FastADK to monitor token usage and costs.
"""

import asyncio
import logging

from fastadk.core.agent import Agent, BaseAgent, tool
from fastadk.tokens import estimate_tokens_and_cost

# Configure logging to see token usage
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@Agent(
    model="gpt-3.5-turbo",
    description="An agent that demonstrates token tracking",
    provider="simulated",  # Using simulated provider for testing without requiring API keys
)
class TokenTrackingAgent(BaseAgent):
    """A simple agent demonstrating token tracking features."""

    @tool
    def get_token_count(self, text: str) -> dict:  # Use built-in dict type instead
        """
        Count tokens in a text string using model tokenizers.

        Args:
            text: The text to count tokens for

        Returns:
            Dictionary with token count and estimated cost
        """
        # Get the token count for various models
        results = {}

        # Count for current model
        results[self._model_name] = estimate_tokens_and_cost(text, self._model_name)

        # Add some other models for comparison
        models = ["gpt-4.1", "claude-3.5-sonnet", "gemini-2.5-flash"]
        for model in models:
            if model != self._model_name:
                results[model] = estimate_tokens_and_cost(text, model)

        return results

    @tool
    def get_token_usage_stats(self) -> dict:  # Use built-in dict type instead
        """
        Get the current token usage statistics for this agent session.

        Returns:
            Dictionary with token usage stats
        """
        return super().get_token_usage_stats()


async def main() -> None:
    """Run the token tracking demo."""
    agent = TokenTrackingAgent()

    # First query - simple question
    print("\n=== Query 1: Simple Question ===")
    response = await agent.run("What is the capital of France?")
    print(f"Response: {response}")

    # Get token usage stats after first query
    print("\n=== Token Usage Stats After Query 1 ===")
    stats = await agent.execute_tool("get_token_usage_stats")
    print(f"Session tokens used: {stats.get('session_tokens_used', 'Unknown')}")
    print(f"Session cost: ${stats.get('session_cost', 'Unknown')}")

    # Second query - more complex
    print("\n=== Query 2: More Complex Question ===")
    response = await agent.run(
        "Explain quantum computing in simple terms and give three practical applications."
    )
    print(f"Response: {response}")

    # Get token usage stats after second query
    print("\n=== Token Usage Stats After Query 2 ===")
    stats = await agent.execute_tool("get_token_usage_stats")
    print(f"Session tokens used: {stats.get('session_tokens_used', 'Unknown')}")
    print(f"Session cost: ${stats.get('session_cost', 'Unknown')}")

    # Use the token counting tool with a shorter text
    print("\n=== Token Counting Tool Demo ===")
    short_text = "This is a short text to demonstrate token counting functionality."

    token_counts = await agent.execute_tool("get_token_count", text=short_text)
    print("Token counts for different models:")
    for model, result in token_counts.items():
        print(
            f"- {model}: {result['tokens']} tokens, estimated cost: ${result['cost']:.6f}"
        )

    # Reset token budget
    print("\n=== Resetting Token Budget ===")
    agent.reset_token_budget()
    stats = await agent.execute_tool("get_token_usage_stats")
    print(
        f"Session tokens used after reset: {stats.get('session_tokens_used', 'Unknown')}"
    )
    print(f"Session cost after reset: ${stats.get('session_cost', 'Unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
