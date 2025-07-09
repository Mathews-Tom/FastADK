# Configuration Patterns

This example demonstrates different patterns for managing configuration in FastADK applications, showing various approaches to loading, merging, validating, and using configuration data.

## Features Demonstrated

- Loading configuration from environment variables
- Loading configuration from YAML files
- Merging configuration from multiple sources
- Validating configuration with Pydantic models
- Using configuration hierarchies
- Overriding configuration at runtime

## Prerequisites

To run this example:

```bash
uv add python-dotenv pyyaml
uv run configuration_patterns.py
```

No API key is required as the example works with the simulated provider.

## How It Works

The example demonstrates six different patterns for configuration management:

### Pattern 1: Environment Variables

- Loads configuration from environment variables with a `FASTADK_` prefix
- Automatically converts `FASTADK_APP_NAME` to `app_name` in the configuration
- Demonstrates retrieving and using environment-based configuration

### Pattern 2: YAML Configuration

- Creates and loads a sample YAML configuration file
- Shows a hierarchical configuration structure with providers, agents, and settings
- Demonstrates validating the loaded configuration

### Pattern 3: Merging Configurations

- Combines configuration from multiple sources (YAML and environment variables)
- Implements a deep merge strategy for nested configuration objects
- Shows how to prioritize one configuration source over another

### Pattern 4: Runtime Configuration Updates

- Updates configuration values dynamically during runtime
- Supports dot notation for accessing nested configuration properties
- Demonstrates safe handling of configuration changes

### Pattern 5: Validating Configuration

- Uses Pydantic models to validate configuration structure and values
- Implements custom validators for specific fields
- Shows error handling for invalid configuration

### Pattern 6: Loading from fastadk.yaml

- Demonstrates loading from a standard configuration file location
- Shows automatic discovery of configuration files
- Provides a consistent location for application settings

## Configuration Models

The example includes several Pydantic models for configuration validation:

- `ProviderConfig`: Provider-specific settings like API keys and timeouts
- `LoggingConfig`: Logging configuration including level, format, and file output
- `MemoryConfig`: Memory backend settings for agent memory
- `AgentConfig`: Agent-specific settings like model, provider, and temperature
- `AppConfig`: Root configuration model containing all other settings

## Expected Output

When you run the script, you'll see output explaining the different configuration patterns:

```bash
============================================================
⚙️  FastADK Configuration Patterns Demo
============================================================

📌 PATTERN 1: ENVIRONMENT VARIABLES
------------------------------------------------------------
Configuration from environment variables:
  app_name: EnvConfigDemo
  environment: development
  logging_level: INFO

Agent configuration:
  Model: gemini-1.5-pro
  Provider: gemini

📌 PATTERN 2: YAML CONFIGURATION
------------------------------------------------------------
Created sample YAML configuration at: /tmp/fastadk_config.yaml

Loaded YAML configuration successfully

Application: ConfigDemoApp v1.0.0
Environment: development

Configured providers: 2
  - openai (timeout: 30s, retries: 3)
  - gemini (timeout: 60s, retries: 2)

Configured agents: 2
  - TextAnalysisAgent: gpt-4-turbo @ openai
    Memory backend: in_memory
  - CreativeWritingAgent: gemini-1.5-pro @ gemini
    Memory backend: vector

📌 PATTERN 3: MERGING CONFIGURATIONS
------------------------------------------------------------
Merged configuration:
  App name: EnvConfigDemo
  Environment: development
  Logging level: INFO

📌 PATTERN 4: RUNTIME CONFIGURATION UPDATES
------------------------------------------------------------
Updated configuration: Updated logging.level to DEBUG
Updated configuration: Updated timeout to 45

Updated configuration values:
  Logging level: DEBUG
  Provider timeout: 45s

📌 PATTERN 5: VALIDATING CONFIGURATION
------------------------------------------------------------
✅ Valid configuration passed validation
  App name: ValidConfigDemo
  Version: 1.0.0
  Environment: development

❌ Invalid configuration failed validation
  (See logs for validation error details)

📌 PATTERN 6: LOADING FROM FASTADK.YAML
------------------------------------------------------------
Created fastadk.yaml at: /tmp/fastadk.yaml

Loaded configuration from fastadk.yaml:
  App name: FastADKDemo
  Version: 1.0.0
  Environment: development

============================================================
🏁 FastADK - Configuration Patterns Demo Completed
============================================================
```

## Key Concepts

1. **Pydantic Validation**: Using Pydantic models to enforce configuration schema and validate values.

2. **Configuration Hierarchy**: Organizing configuration in a logical hierarchy for better management.

3. **Multiple Sources**: Loading configuration from multiple sources and merging them.

4. **Environment-Based Configuration**: Using environment variables for deployment-specific settings.

5. **Runtime Updates**: Safely modifying configuration during application runtime.

## Best Practices Demonstrated

- Using environment variables for sensitive information like API keys
- Validating configuration before using it
- Providing reasonable defaults for optional configuration
- Supporting multiple configuration sources
- Using a standardized configuration file format (YAML)
- Implementing robust error handling for configuration loading
- Sanitizing sensitive information when displaying configuration
