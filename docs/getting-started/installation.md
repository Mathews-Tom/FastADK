# Installation

This guide will help you install FastADK and set up your development environment.

## Prerequisites

Before installing FastADK, make sure you have:

- Python 3.10 or higher
- pip (Python package installer)
- A Google ADK API key (for using Google's Agent Development Kit)

## Installing with pip

The simplest way to install FastADK is using pip:

```bash
pip install fastadk
```

For a development environment, you may want to install the package with extra features:

```bash
pip install fastadk[dev]  # Includes development tools
pip install fastadk[test]  # Includes testing dependencies
pip install fastadk[docs]  # Includes documentation tools
pip install fastadk[all]   # Includes all extras
```

## Using uv (Recommended)

FastADK recommends using [uv](https://github.com/astral-sh/uv) for faster, more reliable package management:

```bash
# Install uv if you don't have it
pip install uv

# Install FastADK with uv
uv pip install fastadk
```

For development:

```bash
# Clone the repository
git clone https://github.com/aetherforge/fastadk.git
cd fastadk

# Create and activate a virtual environment
uv venv

# Install development dependencies
uv pip sync --dev
```

## Setting Up Environment Variables

FastADK uses environment variables for configuration. You can set these in your shell or in a `.env` file in your project directory.

```bash
# Required for using Google ADK
export GOOGLE_API_KEY=your-api-key-here

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
fastadk --version
```

This should display the version number of FastADK.

## Next Steps

Now that you have FastADK installed, you can:

- Continue to the [Quick Start Guide](quick-start.md) to create your first agent
- Explore the [Examples](../examples/weather-agent.md) to see FastADK in action
- Read the [API Reference](../api/core/agent.md) for detailed documentation
