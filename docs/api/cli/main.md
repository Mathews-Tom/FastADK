# CLI Reference

FastADK provides a command-line interface (CLI) for running agents, serving API endpoints, and managing your projects.

## Main Commands

::: fastadk.cli.main.app

## Usage Examples

### Running an Agent

```bash
# Run an agent in interactive mode
fastadk run weather_agent.py

# Run with a specific agent class
fastadk run multi_agent.py --agent-name WeatherAgent
```

### Serving as an API

```bash
# Start an HTTP server with your agent
fastadk serve weather_agent.py

# Specify host and port
fastadk serve weather_agent.py --host 0.0.0.0 --port 8080
```

### Getting Help

```bash
# Display general help
fastadk --help

# Get help for a specific command
fastadk run --help
```

## Environment Variables

The CLI respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FASTADK_ENV` | Environment mode (development, production) | development |
| `FASTADK_LOG_LEVEL` | Logging level | INFO |
| `FASTADK_CONFIG_PATH` | Path to config file | ./fastadk.yaml |
