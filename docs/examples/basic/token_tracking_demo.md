# Token Tracking Demo

This example demonstrates how to use FastADK's token tracking functionality to monitor token usage and costs when interacting with language models.

## Features Demonstrated

- Tracking token usage across agent interactions
- Estimating costs for different language models
- Comparing token counts across multiple providers
- Resetting token budget for session management
- Using the simulated provider for testing without API keys

## Prerequisites

No API keys or additional dependencies are required for this example since it uses the simulated provider.

To run the example:

```bash
uv run token_tracking_demo.py
```

## How It Works

The `TokenTrackingAgent` demonstrates token tracking by:

1. Automatically counting tokens for all prompts and completions
2. Providing tools to access token usage statistics
3. Estimating costs based on model-specific pricing

The example includes two main tools:

1. `get_token_count`: Counts tokens in a provided text string across multiple models
   - Shows token count differences between models
   - Provides cost estimates for each model

2. `get_token_usage_stats`: Returns the current session's token usage statistics
   - Total tokens used in the session
   - Estimated cost of the session
   - Breakdown by prompt and completion tokens

The example also demonstrates how to reset the token budget to start fresh counting.

## Expected Output

When you run the script, you should see output similar to:

```bash
=== Query 1: Simple Question ===
2023-07-09 12:30:45,123 - fastadk.tokens - INFO - Token usage: TokenUsage(prompt=6, completion=10, total=16, model=gpt-3.5-turbo, provider=simulated), estimated cost: $0.000032
Response: Stub response to: What is the capital of France?

=== Token Usage Stats After Query 1 ===
Session tokens used: 16
Session cost: $0.000032

=== Query 2: More Complex Question ===
2023-07-09 12:30:45,234 - fastadk.tokens - INFO - Token usage: TokenUsage(prompt=11, completion=10, total=21, model=gpt-3.5-turbo, provider=simulated), estimated cost: $0.000042
Response: Stub response to: Explain quantum computing in simple terms and give three practical applications.

=== Token Usage Stats After Query 2 ===
Session tokens used: 37
Session cost: $0.000074

=== Token Counting Tool Demo ===
Token counts for different models:
- gpt-3.5-turbo: 11 tokens, estimated cost: $0.000022
- gpt-4.1: 11 tokens, estimated cost: $0.000330
- claude-3.5-sonnet: 16 tokens, estimated cost: $0.000048
- gemini-2.5-flash: 16 tokens, estimated cost: $0.000016

=== Resetting Token Budget ===
Session tokens used after reset: 0
Session cost: $0.0
```

## Key Concepts

1. **Automatic Token Tracking**: FastADK automatically tracks token usage for all agent interactions, including both prompt and completion tokens.

2. **Cost Estimation**: The framework provides cost estimates based on the current pricing of each model.

3. **Cross-Model Comparison**: The example shows how token counts can vary between different models and providers.

4. **Token Budget Management**: The `reset_token_budget()` method demonstrates how to clear token counts for new sessions or budget periods.

5. **Simulated Provider**: The example uses the simulated provider for testing without requiring real API keys.

## Best Practices Demonstrated

- Monitoring token usage to control costs
- Using the simulated provider for testing and development
- Comparing token counts across models before committing to a specific provider
- Resetting token budgets for proper session management
- Logging token usage for audit and analysis

## Practical Applications

Token tracking is useful for:

- Cost estimation for production applications
- Budget enforcement for multi-user systems
- Optimizing prompts to reduce token usage
- Comparing efficiency across different models
- Identifying potential cost savings in agent implementation
