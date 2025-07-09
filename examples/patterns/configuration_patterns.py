"""
Configuration Patterns for FastADK.

This example demonstrates different patterns for managing configuration:
1. Loading configuration from environment variables
2. Loading configuration from YAML files
3. Merging configuration from multiple sources
4. Validating configuration with Pydantic
5. Using configuration hierarchies
6. Overriding configuration at runtime

Usage:
    1. Run the example:
        uv run examples/patterns/configuration_patterns.py
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from fastadk import Agent, BaseAgent, tool

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Try to import load_config, but don't fail if it's not available
# Define a simple load_config function
def load_config(config_file=None):
    """Load configuration from a YAML file."""
    if not config_file:
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Error loading config: %s", e)
        return {}


# ======= CONFIGURATION MODELS =======


class ProviderConfig(BaseModel):
    """Model for provider configuration."""

    name: str
    api_key_env: str
    base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3

    @classmethod
    @field_validator("timeout")
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is within reasonable bounds."""
        if v < 1:
            raise ValueError("Timeout must be at least 1 second")
        if v > 300:
            raise ValueError("Timeout cannot exceed 300 seconds")
        return v


class LoggingConfig(BaseModel):
    """Model for logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: Optional[str] = None

    @classmethod
    @field_validator("level")
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of: {valid_levels}")
        return v


class MemoryConfig(BaseModel):
    """Model for memory configuration."""

    backend: str = "in_memory"
    ttl: Optional[int] = None
    vector_dimensions: int = 1536
    redis_url: Optional[str] = None

    @classmethod
    @field_validator("backend")
    def validate_backend(cls, v: str) -> str:
        """Validate memory backend."""
        valid_backends = ["in_memory", "redis", "vector"]
        v = v.lower()
        if v not in valid_backends:
            raise ValueError(
                f"Invalid memory backend. Must be one of: {valid_backends}"
            )
        return v


class AgentConfig(BaseModel):
    """Model for agent configuration."""

    name: str
    description: str
    model: str
    provider: str
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @classmethod
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within bounds."""
        if v < 0.0:
            raise ValueError("Temperature must be at least 0.0")
        if v > 2.0:
            raise ValueError("Temperature cannot exceed 2.0")
        return v


class AppConfig(BaseModel):
    """Root configuration model."""

    app_name: str
    version: str
    environment: str = "development"
    providers: List[ProviderConfig]
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agents: List[AgentConfig]

    @classmethod
    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_environments = ["development", "testing", "production"]
        v = v.lower()
        if v not in valid_environments:
            raise ValueError(
                f"Invalid environment. Must be one of: {valid_environments}"
            )
        return v


# ======= CONFIGURATION HELPERS =======


def create_sample_yaml_config() -> str:
    """Create a sample YAML configuration file."""
    config_yaml = """
# FastADK Application Configuration

app_name: ConfigDemoApp
version: 1.0.0
environment: development

# Provider configurations
providers:
  - name: openai
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
    timeout: 30
    retry_attempts: 3

  - name: gemini
    api_key_env: GEMINI_API_KEY
    timeout: 60
    retry_attempts: 2

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_to_file: false
  log_file: app.log

# Agent configurations
agents:
  - name: TextAnalysisAgent
    description: An agent that analyzes text content
    model: gpt-4-turbo
    provider: openai
    temperature: 0.3
    system_prompt: You are an expert text analysis assistant.
    memory:
      backend: in_memory
      ttl: 3600

  - name: CreativeWritingAgent
    description: An agent that generates creative content
    model: gemini-1.5-pro
    provider: gemini
    temperature: 0.8
    system_prompt: You are a creative writing assistant with a vivid imagination.
    memory:
      backend: vector
      vector_dimensions: 1536
"""

    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    config_path = Path(temp_dir) / "fastadk_config.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)

    return str(config_path)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error("Error loading YAML config: %s", e)
        return {}


def load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    # Look for environment variables with prefix FASTADK_
    env_config = {}

    for key, value in os.environ.items():
        if key.startswith("FASTADK_"):
            # Convert FASTADK_APP_NAME to app_name
            config_key = key[8:].lower()
            env_config[config_key] = value

    return env_config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}

    for config in configs:
        # Deep merge nested dictionaries
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> Optional[AppConfig]:
    """Validate configuration using Pydantic model."""
    try:
        app_config = AppConfig(**config)
        return app_config
    except ValidationError as e:
        logger.error("Configuration validation error: %s", e)
        return None


# ======= CONFIGURATION DEMO AGENT =======


@Agent(
    model="gemini-1.5-pro",
    description="An agent demonstrating configuration patterns",
    provider="gemini",  # Will fall back to simulated if no API key is available
)
class ConfigDemoAgent(BaseAgent):
    """Agent demonstrating different configuration patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # We'll initialize the agent with a custom config if provided
        self.config = config or {}
        super().__init__()

    @tool
    def get_current_config(self) -> dict:
        """
        Get the current configuration.

        Returns:
            Current configuration settings
        """
        # Return a sanitized version of the config (without API keys)
        sanitized_config = {}

        if hasattr(self, "config") and self.config:
            sanitized_config = self._sanitize_config(self.config)

        # Add agent configuration - access attributes safely
        sanitized_config["agent"] = {
            "model": getattr(self, "model", "unknown"),
            "provider": getattr(self, "provider", "unknown"),
            "description": getattr(
                self, "description", getattr(self, "_description", "unknown")
            ),
        }

        return sanitized_config

    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from configuration."""
        result = {}

        for key, value in config.items():
            # Skip API keys and other sensitive fields
            if any(
                sensitive in key.lower()
                for sensitive in ["api_key", "password", "secret", "token"]
            ):
                result[key] = "********"
            elif isinstance(value, dict):
                result[key] = self._sanitize_config(value)
            else:
                result[key] = value

        return result

    @tool
    def update_config(self, key_path: str, value: Any) -> dict:
        """
        Update a configuration value.

        Args:
            key_path: Dot-notation path to config key (e.g., "logging.level")
            value: New value

        Returns:
            Result of the operation
        """
        if not hasattr(self, "config"):
            self.config = {}

        # Parse key path
        keys = key_path.split(".")

        # Navigate to the correct level
        current = self.config
        for key in keys[:-1]:
            # Handle list indices in the format "list_name.0.property"
            if "." in key and key.split(".")[-1].isdigit():
                list_name, index = key.split(".", 1)
                index = int(index)
                if list_name not in current:
                    current[list_name] = []
                while len(current[list_name]) <= index:
                    current[list_name].append({})
                current = current[list_name][index]
            else:
                if key not in current:
                    current[key] = {}
                # Check if we're dealing with a list
                if isinstance(current[key], list) and keys[-1].isdigit():
                    index = int(keys[-1])
                    if index >= len(current[key]):
                        # Extend the list if needed
                        current[key].extend([{}] * (index - len(current[key]) + 1))
                current = current[key]

        # Update the value
        current[keys[-1]] = value

        return {
            "success": True,
            "message": f"Updated {key_path} to {value}",
            "updated_config": self._sanitize_config(self.config),
        }

    @tool
    def load_config_from_yaml(self, file_path: str) -> dict:
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Result of the operation
        """
        try:
            yaml_config = load_yaml_config(file_path)

            if not yaml_config:
                return {
                    "success": False,
                    "message": f"Failed to load configuration from {file_path}",
                }

            # Validate the configuration
            app_config = validate_config(yaml_config)

            if not app_config:
                return {
                    "success": False,
                    "message": "Configuration validation failed",
                }

            # Update agent config
            self.config = yaml_config

            return {
                "success": True,
                "message": f"Loaded configuration from {file_path}",
                "config": self._sanitize_config(yaml_config),
            }
        except Exception as e:
            logger.error("Error loading YAML config: %s", e)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
            }


async def demonstrate_configuration_patterns() -> None:
    """Run the configuration patterns demonstration."""
    print("\n" + "=" * 60)
    print("⚙️  FastADK Configuration Patterns Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    try:
        # PATTERN 1: ENVIRONMENT VARIABLES
        print("\n\n📌 PATTERN 1: ENVIRONMENT VARIABLES")
        print("-" * 60)

        # Set some environment variables for demo
        os.environ["FASTADK_APP_NAME"] = "EnvConfigDemo"
        os.environ["FASTADK_ENVIRONMENT"] = "development"
        os.environ["FASTADK_LOGGING_LEVEL"] = "INFO"

        # Load configuration from environment variables
        env_config = load_env_config()

        print("Configuration from environment variables:")
        for key, value in env_config.items():
            print(f"  {key}: {value}")

        # Create agent with environment config
        agent = ConfigDemoAgent(config=env_config)

        # Get current config
        result = await agent.execute_tool("get_current_config")
        print("\nAgent configuration:")
        print(f"  Model: {result.get('agent', {}).get('model', 'unknown')}")
        print(f"  Provider: {result.get('agent', {}).get('provider', 'unknown')}")

        # PATTERN 2: YAML CONFIGURATION
        print("\n\n📌 PATTERN 2: YAML CONFIGURATION")
        print("-" * 60)

        # Create a sample YAML config file
        yaml_path = create_sample_yaml_config()
        print(f"Created sample YAML configuration at: {yaml_path}")

        # Load configuration from YAML
        result = await agent.execute_tool("load_config_from_yaml", file_path=yaml_path)

        if result.get("success", False):
            print("\nLoaded YAML configuration successfully")

            # Display some key configuration values
            config = result.get("config", {})
            print(f"\nApplication: {config.get('app_name')} v{config.get('version')}")
            print(f"Environment: {config.get('environment')}")

            # Display provider configuration
            providers = config.get("providers", [])
            print(f"\nConfigured providers: {len(providers)}")
            for provider in providers:
                print(
                    f"  - {provider['name']} (timeout: {provider['timeout']}s, retries: {provider['retry_attempts']})"
                )

            # Display agent configurations
            agents = config.get("agents", [])
            print(f"\nConfigured agents: {len(agents)}")
            for agent_config in agents:
                print(
                    f"  - {agent_config['name']}: {agent_config['model']} @ {agent_config['provider']}"
                )
                print(f"    Memory backend: {agent_config['memory']['backend']}")
        else:
            print(f"\nFailed to load YAML configuration: {result.get('message')}")

        # PATTERN 3: MERGING CONFIGURATIONS
        print("\n\n📌 PATTERN 3: MERGING CONFIGURATIONS")
        print("-" * 60)

        # Load both configs
        yaml_config = load_yaml_config(yaml_path)

        # Merge configurations (environment variables take precedence)
        merged_config = merge_configs(yaml_config, env_config)

        print("Merged configuration:")
        print(f"  App name: {merged_config.get('app_name')}")
        print(f"  Environment: {merged_config.get('environment')}")
        print(f"  Logging level: {merged_config.get('logging', {}).get('level')}")

        # PATTERN 4: RUNTIME CONFIGURATION UPDATES
        print("\n\n📌 PATTERN 4: RUNTIME CONFIGURATION UPDATES")
        print("-" * 60)

        # Update configuration at runtime
        result = await agent.execute_tool(
            "update_config", key_path="logging.level", value="DEBUG"
        )

        if result.get("success", False):
            print(f"Updated configuration: {result.get('message')}")

        # Use a simpler update for demo
        result = await agent.execute_tool("update_config", key_path="timeout", value=45)

        if result.get("success", False):
            print(f"Updated configuration: {result.get('message')}")

        # Get current config
        result = await agent.execute_tool("get_current_config")
        updated_config = result

        print("\nUpdated configuration values:")

        try:
            logging_level = updated_config.get("logging", {}).get("level")
            if logging_level:
                print(f"  Logging level: {logging_level}")

            provider_timeout = updated_config.get("providers", [{}])[0].get("timeout")
            if provider_timeout:
                print(f"  Provider timeout: {provider_timeout}s")
        except (IndexError, KeyError):
            print("  Could not retrieve updated values")

        # PATTERN 5: VALIDATING CONFIGURATION
        print("\n\n📌 PATTERN 5: VALIDATING CONFIGURATION")
        print("-" * 60)

        # Valid configuration
        valid_config = {
            "app_name": "ValidConfigDemo",
            "version": "1.0.0",
            "environment": "development",
            "providers": [
                {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout": 30,
                    "retry_attempts": 3,
                }
            ],
            "agents": [
                {
                    "name": "TextAgent",
                    "description": "Text processing agent",
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.7,
                }
            ],
        }

        app_config = validate_config(valid_config)

        if app_config:
            print("✅ Valid configuration passed validation")
            print(f"  App name: {app_config.app_name}")
            print(f"  Version: {app_config.version}")
            print(f"  Environment: {app_config.environment}")

        # Invalid configuration
        invalid_config = {
            "app_name": "InvalidConfigDemo",
            "version": "1.0.0",
            "environment": "invalid_env",  # Invalid environment
            "providers": [
                {
                    "name": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout": 0,  # Invalid timeout
                    "retry_attempts": 3,
                }
            ],
            "agents": [
                {
                    "name": "TextAgent",
                    "description": "Text processing agent",
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 3.0,  # Invalid temperature
                }
            ],
        }

        app_config = validate_config(invalid_config)

        if not app_config:
            print("\n❌ Invalid configuration failed validation")
            print("  (See logs for validation error details)")

        # PATTERN 6: LOADING FROM FASTADK.YAML
        print("\n\n📌 PATTERN 6: LOADING FROM FASTADK.YAML")
        print("-" * 60)

        # Create a temporary fastadk.yaml file
        temp_dir = tempfile.gettempdir()
        fastadk_config_path = Path(temp_dir) / "fastadk.yaml"

        # Write a minimal config
        with open(fastadk_config_path, "w", encoding="utf-8") as f:
            f.write(
                """
app_name: FastADKDemo
version: 1.0.0
environment: development
providers:
  - name: openai
    api_key_env: OPENAI_API_KEY
  - name: gemini
    api_key_env: GEMINI_API_KEY
agents:
  - name: DemoAgent
    description: A demo agent
    model: gpt-4
    provider: openai
            """
            )

        print(f"Created fastadk.yaml at: {fastadk_config_path}")

        # Load using the built-in config loader
        try:
            # Save current directory to restore it later
            original_dir = os.getcwd()
            os.chdir(temp_dir)

            # Load configuration using the local YAML file
            with open("fastadk.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            print("\nLoaded configuration from fastadk.yaml:")
            print(f"  App name: {config.get('app_name')}")
            print(f"  Version: {config.get('version')}")
            print(f"  Environment: {config.get('environment')}")

            # Restore original directory
            os.chdir(original_dir)
        except Exception as e:
            print(f"\nError loading fastadk.yaml: {e}")

        # Clean up
        try:
            os.remove(yaml_path)
            os.remove(fastadk_config_path)
        except Exception:
            pass

        print("\n" + "=" * 60)
        print("🏁 FastADK - Configuration Patterns Demo Completed")
        print("=" * 60)
    except Exception as e:
        logger.error("Error in configuration patterns demo: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")


async def main() -> None:
    """Run the main demo."""
    await demonstrate_configuration_patterns()


if __name__ == "__main__":
    asyncio.run(main())
