"""
LiteLLM Provider Demo

This example demonstrates how to use FastADK with LiteLLM as a provider.
LiteLLM gives you a unified interface to 100+ LLM APIs from OpenAI,
Anthropic, Cohere, Hugging Face, and others.

Usage:
    1. Install FastADK and LiteLLM: `uv add litellm`
    2. Set your LiteLLM API key: `export LITELLM_API_KEY=your-api-key`
    3. Run this example: `uv run -m examples.basic.litellm_demo`

You can use LiteLLM in two modes:
- SDK mode: Uses the LiteLLM Python package directly
- Proxy mode: Uses the LiteLLM proxy server for more features like routing,
                fallbacks, caching, etc.

This example demonstrates SDK mode, which is the simplest to get started with.
"""

import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool

# Load environment variables from .env file
load_dotenv()


@Agent(
    model="gpt-4.1",  # You can use any model supported by LiteLLM
    provider="litellm",
    description="An agent that uses LiteLLM as a provider",
)
class LiteLLMAgent(BaseAgent):
    """A simple agent that uses LiteLLM as a provider."""

    @tool
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get the current weather for a location."""
        print(f"Getting weather for {location}...")
        # In a real implementation, this would call a weather API
        return {
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 40,
        }


async def main() -> None:
    """Run the LiteLLM agent example."""
    # Check if the API key is set
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: LITELLM_API_KEY or OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or export it in your terminal.")
        sys.exit(1)

    # Create the agent
    agent = LiteLLMAgent()

    # Run the agent with a prompt
    print("\n🤖 Sending a question to the LiteLLM agent...\n")
    response = await agent.run(
        "What's the weather like in San Francisco? Can you also tell me what makes LiteLLM special?"
    )

    # Print the response
    print(f"\n✨ Agent response:\n{response}")

    # Display token usage stats if available
    token_stats = agent.get_token_usage_stats()
    if token_stats.get("token_tracking_enabled", False) is not False:
        print("\n📊 Token usage statistics:")
        print(
            f"  - Session tokens used: {token_stats.get('session_tokens_used', 'N/A')}"
        )
        print(f"  - Session cost: ${token_stats.get('session_cost', 'N/A'):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
