# LiteLLM Provider Demo

This example demonstrates how to use FastADK with LiteLLM as a provider, giving you access to 100+ LLM APIs from OpenAI, Anthropic, Cohere, Hugging Face, and others through a unified interface.

## Features Demonstrated

- Using LiteLLM as a provider for FastADK
- Accessing different language models through a unified interface
- Token usage tracking with cost estimation
- Environment variable configuration for API keys
- Graceful error handling for missing credentials

## Prerequisites

To run this example, you need:

1. Install the LiteLLM package:

   ```bash
   uv add litellm python-dotenv
   ```

2. Set your API key either via environment variable:

   ```bash
   export LITELLM_API_KEY=your_api_key_here
   # OR use a specific provider key
   export OPENAI_API_KEY=your_api_key_here
   ```

   Or in a `.env` file:

   ```env
   LITELLM_API_KEY=your_api_key_here
   # OR
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the example:

   ```bash
   uv run litellm_demo.py
   ```

## How It Works

The example creates a `LiteLLMAgent` with the following characteristics:

1. Uses `provider="litellm"` to specify LiteLLM as the provider
2. Configures the agent to use "gpt-4.1" model (which can be any model supported by LiteLLM)
3. Implements a simple weather tool to demonstrate tool integration
4. Tracks token usage to show costs

When run, the agent processes a query about both the weather in San Francisco and what makes LiteLLM special, showing how the agent can combine custom tool data with model-generated information.

## LiteLLM Modes

LiteLLM can be used in two modes:

1. **SDK Mode** (demonstrated in this example): Uses the LiteLLM Python package directly, which is the simplest way to get started.

2. **Proxy Mode**: Uses the LiteLLM proxy server for advanced features like:
   - Request routing
   - Model fallbacks
   - Response caching
   - Load balancing
   - Detailed usage tracking

## Expected Output

When you run the script, you should see output similar to:

```bash
🤖 Sending a question to the LiteLLM agent...

✨ Agent response:
The weather in San Francisco is currently sunny with a temperature of 72°F and 40% humidity.

LiteLLM is special because it provides a unified interface to access over 100 different LLM APIs from various providers like OpenAI, Anthropic, Cohere, Hugging Face, and many others. This means you can easily switch between different language models without changing your code. It also offers features like routing, fallbacks, caching, and more. Essentially, it simplifies working with multiple LLM providers through a single, consistent API.

📊 Token usage statistics:
  - Session tokens used: 210
  - Session cost: $0.000630
```

## Key Concepts

1. **Provider Flexibility**: LiteLLM allows you to switch between different LLM providers by simply changing the model name, without modifying your code.

2. **Unified API**: The same code can work with models from OpenAI, Anthropic, Cohere, and many other providers.

3. **Token Tracking**: FastADK's token tracking works seamlessly with LiteLLM, providing cost estimates across different providers.

4. **Environment Variables**: The example shows how to use environment variables to securely store API keys.

## Best Practices Demonstrated

- Using environment variables for API key management
- Checking for required credentials before attempting to use the agent
- Providing clear error messages when credentials are missing
- Displaying token usage for cost monitoring
- Using a unified provider interface for maximum flexibility

## Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Models](https://docs.litellm.ai/docs/providers)
- [FastADK Provider Documentation](https://fastadk.org/providers/litellm/)
