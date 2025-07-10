# FastADK Examples Index

This document provides an organized index of all examples available in the FastADK repository, making it easy to find the right example for your needs.

## Basic Examples

These examples demonstrate fundamental FastADK concepts and are a great starting point:

- [**Weather Agent**](examples/basic/weather_agent.md) - Core agent functionality with real API integration (wttr.in)
- [**Exception Handling**](examples/basic/exception_demo.md) - Comprehensive exception handling and error management
- [**Token Tracking**](examples/basic/token_tracking_demo.md) - Token usage tracking and cost estimation
- [**LiteLLM Integration**](examples/basic/litellm_demo.md) - Integration with LiteLLM for provider flexibility
- [**Reasoning Demo**](examples/basic/reasoning_demo.md) - Chain-of-thought reasoning with visible tool selection

## Advanced Examples

These examples showcase more complex usage patterns and advanced features:

- [**Travel Assistant**](examples/advanced/travel_assistant.py) - Comprehensive example with memory, tools, API integration, lifecycle hooks
- [**Workflow Demo**](examples/advanced/workflow_demo.md) - Workflow orchestration with sequential/parallel flows
- [**Batch Processing**](examples/advanced/batch_processing_demo.py)- Efficient batch processing of multiple inputs
- [**Multi-Provider Reasoning**](examples/advanced/multi_provider_reasoning.py) - Using multiple providers based on available API keys
- [**Customer Support**](examples/advanced/customer_support.md) - Building a customer support assistant
- [**Finance Assistant**](examples/advanced/finance_assistant.md) - Financial data analysis and reporting assistant

## API Examples

These examples demonstrate integrating FastADK with APIs:

- [**HTTP Agent**](examples/api/http_agent.md) - Serving agents via HTTP API with FastAPI

## UI Examples

These examples show how to build user interfaces for FastADK agents:

- [**Streamlit Chat App**](examples/ui/streamlit_chat_app.md) - Building interactive chat interfaces with Streamlit

## Pattern Examples

These examples demonstrate recommended patterns and practices:

- [**Tool Patterns**](examples/patterns/tool_patterns.md) - Different tool development patterns (async/sync, validation)
- [**Configuration Patterns**](examples/patterns/configuration_patterns.md) - Configuration loading from YAML, environment, etc.

## Training Examples

These examples focus on model training and fine-tuning:

- [**Fine Tuning Example**](examples/training/fine_tuning_example.md) - Data format conversion and fine-tuning jobs

## Getting Started

To run these examples:

1. Clone the FastADK repository
2. Install dependencies: `uv add -e .`
3. Set up environment variables in `.env` (see `.env.example` in the examples directory)
4. Run an example: `uv run python examples/basic/weather_agent.py`

## Contributing

If you'd like to contribute a new example, please follow our [contribution guidelines](docs/contributing/guidelines.md) and ensure your example follows the established patterns.
