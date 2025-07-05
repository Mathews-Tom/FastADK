# FastADK Examples

This directory contains examples demonstrating various features and capabilities of the FastADK framework.

## Directory Structure

- **basic/**: Simple examples showing core functionality
  - `weather_agent.py`: A basic weather agent with API integration
  - `exception_demo.py`: Demonstrates exception handling features

- **advanced/**: More complex examples with advanced features
  - `travel_assistant.py`: Comprehensive travel assistant with memory, tools, and lifecycle hooks
  - `workflow_demo.py`: Demonstrates workflow orchestration capabilities

- **api/**: Examples showing API integration
  - `http_agent.py`: Multi-agent HTTP API server with FastAPI

## Running the Examples

Each example includes detailed setup instructions in its header comments and the README.md file in its directory.

### Environment Setup

Most examples can run with minimal setup, but here are the general requirements:

1. **Required Packages**:

   ```bash
   # For basic examples
   uv add httpx requests
   
   # For advanced examples
   uv add httpx python-dotenv
   
   # For API examples
   uv add fastapi uvicorn
   ```

2. **API Keys**:

   For examples using language models, you'll need to set up API keys using either:

   **Environment variables**:

   ```bash
   # For Gemini (used in most examples)
   export GEMINI_API_KEY=your_api_key_here
   
   # Optional for multi-provider examples
   export OPENAI_API_KEY=your_api_key_here
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

   **Or a .env file** in the project root directory:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_api_key_here
   ANTHROPIC_API_KEY=your_api_key_here
   ```

   (This requires python-dotenv: `uv add python-dotenv`)

3. **Running an Example**:

   ```bash
   # Navigate to the FastADK directory
   cd /path/to/FastADK
   
   # Run an example
   uv run examples/basic/weather_agent.py
   ```

## Example Descriptions

### Basic Examples

- **Weather Agent**: A simple agent that fetches real weather data from wttr.in and provides forecasts and current conditions.
- **Exception Demo**: Demonstrates the comprehensive exception handling system in FastADK.

### Advanced Examples

- **Travel Assistant**: A comprehensive travel assistant that demonstrates:
  - Multiple tool implementations (flights, hotels, weather, currency)
  - Memory for tracking user preferences
  - Lifecycle hooks for metrics
  - Error handling and fallbacks
  - API integrations

- **Workflow Demo**: Demonstrates FastADK's workflow capabilities:
  - Sequential and parallel workflows
  - Conditional branching
  - Error handling and retries
  - Data transformation and merging

### API Examples

- **HTTP Agent**: Shows how to:
  - Create multiple agent classes
  - Register them with FastADK's registry
  - Serve them via HTTP API with FastAPI
  - Use different LLM providers (Gemini, OpenAI, Anthropic)

## Example Output

Each example directory contains a README.md file with sample output showing what to expect when running the examples.
