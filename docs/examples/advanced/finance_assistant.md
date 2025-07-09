# Finance Assistant Example

## Overview

This example demonstrates a comprehensive financial assistant agent built with FastADK. The agent can provide financial advice, stock information, and calculate various financial metrics like compound interest, mortgage payments, taxes, and retirement projections.

## Features Demonstrated

- **Token Budget Management**: Setting and enforcing token limits for cost control
- **Complex Tool Implementations**: Multiple financial calculation tools with proper validation
- **Error Handling**: Robust error handling in financial calculations
- **JSON Response Formatting**: Structured data responses for financial calculations
- **Memory Management**: Using InMemoryBackend for conversation persistence

## Running this Example

```bash
uv run examples/advanced/finance_assistant.py
```

## Example Interactions

### Checking Stock Prices

```bash
You: What's the current price of Apple stock?
Assistant: The current price of Apple Inc. (AAPL) is $185.92 (+0.75%)
```

### Calculating Compound Interest

```bash
You: Calculate compound interest on a $10,000 investment at 5% annual interest for 10 years
Assistant: [Provides detailed breakdown of compound interest calculation with yearly results]
```

### Mortgage Payment Calculation

```bash
You: What would be my monthly payment for a $300,000 mortgage at 4.5% interest for 30 years?
Assistant: [Provides monthly payment amount and amortization details]
```

### Tax Estimation

```bash
You: Estimate my taxes if I earn $75,000 per year as a single filer
Assistant: [Provides tax breakdown by brackets with effective tax rate]
```

### Retirement Planning

```bash
You: How much will I have for retirement if I'm 35 now, retire at 65, have $50,000 saved, and contribute $500 monthly with 7% returns?
Assistant: [Provides detailed retirement savings projection]
```

## Implementation Details

The agent is implemented with:

- Multiple specialized financial calculation tools
- Structured JSON responses for clear data presentation
- Extensive error checking and validation
- Token budget settings to control usage

## Customization

You can extend this example by:

- Adding more financial tools (e.g., college savings calculator)
- Connecting to real financial data APIs
- Implementing a user profile system to save financial preferences
- Adding visualization capabilities for financial projections

## Requirements

- fastadk
- Basic Python knowledge
- An OpenAI API key (set as OPENAI_API_KEY environment variable)
