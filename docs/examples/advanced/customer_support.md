# Customer Support Agent Example

## Overview

This example demonstrates a comprehensive customer support agent for a consumer electronics company. The agent can handle product inquiries, order status checks, refund requests, and technical support issues using a knowledge base of common problems and solutions.

## Features Demonstrated

- **Custom Context Policy**: `PrioritySupportContextPolicy` that prioritizes support-related keywords in the conversation
- **Memory Backend**: Using InMemoryBackend for conversation persistence
- **Structured Knowledge Base**: Integration with product catalog and technical solutions
- **Multi-Step Processes**: Order tracking, refund processing, and support ticket creation
- **Domain-Specific Tools**: Tools designed specifically for customer service tasks

## Running this Example

```bash
uv run examples/advanced/customer_support.py
```

## Example Interactions

### Product Information

```bash
You: Tell me about your SmartHome Hub product
Assistant: [Provides detailed product information about the SmartHome Hub]
```

### Order Status Check

```bash
You: What's the status of my order ORD12345?
Assistant: [Provides order details, status, and estimated delivery date]
```

### Technical Support

```bash
You: My SmartHome Hub won't connect to WiFi
Assistant: [Provides troubleshooting steps from knowledge base]
```

### Refund Request

```bash
You: I want to return my Wireless Earbuds and get a refund
Assistant: [Guides through refund process and eligibility]
```

### Support Ticket Creation

```bash
You: I need to create a support ticket for my broken TV
Assistant: [Creates ticket and provides reference number]
```

## Implementation Details

The agent is implemented with:

- A priority-based context policy that emphasizes support-related conversation
- Mock databases for products, orders, customers, and support tickets
- Structured knowledge base for common technical issues
- Comprehensive tool set for different support scenarios

### Context Policy

The `PrioritySupportContextPolicy` demonstrates how to customize conversation context handling to prioritize certain types of messages (those containing keywords like "ticket", "issue", "problem", etc.) when the conversation history exceeds the token limit.

## Customization

You can extend this example by:

- Connecting to real CRM and order management systems
- Adding authentication and customer verification
- Implementing sentiment analysis for customer satisfaction tracking
- Adding live agent handoff for complex issues
- Integrating with email or SMS notification systems

## Requirements

- fastadk
- Basic Python knowledge
- An OpenAI API key (set as OPENAI_API_KEY environment variable)
