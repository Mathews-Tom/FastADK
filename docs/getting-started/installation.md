# Installation

This guide will help you install FastADK and set up your development environment.

## Prerequisites

Before installing FastADK, make sure you have:

- Python 3.10 or higher
- API keys for the LLM providers you plan to use (Gemini, OpenAI, Anthropic, etc.)

## Installing UV (Recommended)

FastADK recommends using [UV](https://github.com/astral-sh/uv), a significantly faster package installer and resolver for Python:

```bash
# Install UV using pip (only needed once)
pip install uv

# On macOS with Homebrew
brew install uv

# On Linux with pipx
pipx install uv
```

## Installing FastADK with UV (Recommended)

Once you have UV installed, you can install FastADK:

```bash
# Basic installation
uv pip install fastadk

# With extras
uv pip install "fastadk[dev]"     # Includes development tools
uv pip install "fastadk[test]"    # Includes testing dependencies
uv pip install "fastadk[docs]"    # Includes documentation tools
uv pip install "fastadk[redis]"   # Includes Redis memory backend
uv pip install "fastadk[vector]"  # Includes vector database support
uv pip install "fastadk[all]"     # Includes all extras
```

## Installing with Pip (Alternative)

If you prefer using pip, you can install FastADK with:

```bash
pip install fastadk

# With extras
pip install "fastadk[dev]"     # Includes development tools
pip install "fastadk[test]"    # Includes testing dependencies
pip install "fastadk[docs]"    # Includes documentation tools
pip install "fastadk[redis]"   # Includes Redis memory backend
pip install "fastadk[vector]"  # Includes vector database support
pip install "fastadk[all]"     # Includes all extras
```

## Development Setup with UV

For setting up a development environment:

```bash
# Clone the repository
git clone https://github.com/aetherforge/fastadk.git
cd fastadk

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install development dependencies
uv pip install -e ".[dev]"
```

## Setting Up Environment Variables

FastADK uses environment variables for configuration. You can set these in your shell or in a `.env` file in your project directory.

```bash
# API keys for different LLM providers
export GEMINI_API_KEY=your-gemini-api-key-here
export OPENAI_API_KEY=your-openai-api-key-here
export ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional configuration
export FASTADK_ENV=development  # Options: development, production, testing
export FASTADK_LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
export FASTADK_MEMORY_BACKEND=inmemory  # Options: inmemory, redis
```

If using Redis for memory:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=optional-password
```

## Verifying Installation

To verify that FastADK is correctly installed, run:

```bash
# Activate your virtual environment if not already activated
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Import and check version
uv run -c "import fastadk; print(f'FastADK version: {fastadk.__version__}')"
```

This should display the version number of FastADK.

## Running Examples

FastADK includes several examples to help you get started. To run them:

```bash
# Clone the repository if you haven't already
git clone https://github.com/aetherforge/fastadk.git
cd fastadk

# Create and activate a virtual environment if you haven't already
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Install FastADK and dependencies
uv pip install -e ".[all]"

# Run a basic example
uv run examples/basic/weather_agent.py

# Run an advanced example
uv run examples/advanced/travel_assistant.py

# Start the HTTP API server example
uv run examples/api/http_agent.py
```

## UV Benefits for FastADK Development

Using UV with FastADK offers several advantages:

- **Speed**: UV installs packages 10-100x faster than pip
- **Reliability**: Better dependency resolution and fewer conflicts
- **Reproducibility**: More consistent installations across environments
- **Efficiency**: Reduced memory usage during package installation
- **Modern**: Latest Python packaging standards support

## Next Steps

Now that you have FastADK installed, you can:

- Continue to the [Quick Start Guide](quick-start.md) to create your first agent
- Explore the [System Overview](../system-overview.md) to understand FastADK's architecture
- Check out the [Examples](../examples/basic/reasoning_demo.md) to see FastADK in action
- Read the [API Reference](../api/core/agent.md) for detailed documentation
