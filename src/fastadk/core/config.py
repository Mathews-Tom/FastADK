"""
Configuration system for FastADK.

This module provides a Pydantic-based configuration system that supports
environment variables, YAML/TOML configuration files, and defaults.
"""

import os
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants
DEFAULT_CONFIG_PATHS = [
    "./fastadk.yaml",
    "./fastadk.yml",
    "./fastadk.toml",
    "~/.fastadk/config.yaml",
    "~/.fastadk/config.yml",
    "~/.fastadk/config.toml",
]


class EnvironmentType(str, Enum):
    """Environment type for FastADK configuration."""

    DEVELOPMENT = "dev"
    PRODUCTION = "prod"
    TESTING = "test"


class MemoryBackendType(str, Enum):
    """Memory backend types supported by FastADK."""

    IN_MEMORY = "inmemory"
    REDIS = "redis"
    FIRESTORE = "firestore"
    VECTOR = "vector"
    CUSTOM = "custom"


class LogLevel(str, Enum):
    """Log levels for FastADK logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelConfig(BaseModel):
    """Configuration for AI models."""

    provider: str = Field(
        default="gemini",
        description="The model provider (gemini, vertex, openai, anthropic, etc.)",
    )
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="The model name to use",
    )
    api_key_env_var: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable containing the API key",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for model API calls in seconds",
        ge=1,
        le=300,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for model API calls",
        ge=0,
        le=10,
    )
    track_tokens: bool = Field(
        default=True,
        description="Whether to track token usage and cost",
    )
    custom_price_per_1k: dict[str, float] = Field(
        default_factory=dict,
        description="Custom price per 1K tokens, with 'input' and 'output' keys",
    )
    additional_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters to pass to the model",
    )

    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, v: str) -> str:
        """Validate that the API key environment variable is set."""
        if v and v not in os.environ:
            print(
                f"Warning: Environment variable {v} is not set. "
                "API calls may fail without a valid API key."
            )
        return v


class ContextPolicyType(str, Enum):
    """Context policy types for managing conversation history."""

    MOST_RECENT = "most_recent"
    SUMMARIZE_OLDER = "summarize_older"
    HYBRID_VECTOR = "hybrid_vector"
    CUSTOM = "custom"


class ContextPolicyConfig(BaseModel):
    """Configuration for context policies."""

    policy_type: ContextPolicyType = Field(
        default=ContextPolicyType.MOST_RECENT,
        description="The type of context policy to use",
    )
    max_messages: int = Field(
        default=10,
        description="Maximum number of messages to include in context",
        ge=1,
    )
    threshold_tokens: int | None = Field(
        default=None,
        description="Token threshold that triggers summarization",
    )
    vector_k: int = Field(
        default=3,
        description="Number of semantically relevant messages to retrieve",
        ge=1,
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for relevant messages",
        ge=0.0,
        le=1.0,
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options for the context policy",
    )


class MemoryConfig(BaseModel):
    """Configuration for memory backends."""

    backend_type: MemoryBackendType = Field(
        default=MemoryBackendType.IN_MEMORY,
        description="The type of memory backend to use",
    )
    connection_string: str | None = Field(
        default=None,
        description="Connection string for the memory backend (if applicable)",
    )
    ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for memory entries in seconds",
        ge=0,
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options for the memory backend",
    )
    context_policy: ContextPolicyConfig = Field(
        default_factory=ContextPolicyConfig,
        description="Configuration for context policy",
    )


class TelemetryConfig(BaseModel):
    """Configuration for telemetry and observability."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable telemetry",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level for FastADK",
    )
    metrics_enabled: bool = Field(
        default=False,
        description="Whether to enable metrics collection",
    )
    tracing_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )
    anonymize_data: bool = Field(
        default=True,
        description="Whether to anonymize telemetry data",
    )


class SecurityConfig(BaseModel):
    """Configuration for security features."""

    content_filtering: bool = Field(
        default=True,
        description="Whether to enable content filtering",
    )
    pii_detection: bool = Field(
        default=False,
        description="Whether to enable PII detection and masking",
    )
    audit_logging: bool = Field(
        default=False,
        description="Whether to enable detailed audit logging",
    )
    max_token_limit: int = Field(
        default=4096,
        description="Maximum token limit for agent responses",
        ge=1,
    )


class TokenBudgetConfig(BaseModel):
    """Configuration for token usage budget."""

    max_tokens_per_request: int | None = Field(
        default=None,
        description="Maximum tokens allowed per request",
    )
    max_tokens_per_session: int | None = Field(
        default=None,
        description="Maximum tokens allowed per session",
    )
    max_cost_per_request: float | None = Field(
        default=None,
        description="Maximum cost allowed per request in USD",
    )
    max_cost_per_session: float | None = Field(
        default=None,
        description="Maximum cost allowed per session in USD",
    )
    warn_at_percent: float = Field(
        default=80.0,
        description="Warn when budget reaches this percentage",
        ge=1.0,
        le=100.0,
    )


class FastADKSettings(BaseSettings):
    """
    Main settings class for FastADK.

    This class follows the precedence order:
    1. Environment variables
    2. Configuration file (yaml/toml)
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="FASTADK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General settings
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="The environment type (dev, prod, test)",
    )
    config_path: str | None = Field(
        default=None,
        description="Path to the configuration file",
    )

    # Component configurations
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory backend configuration",
    )
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Telemetry and observability configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration",
    )
    token_budget: TokenBudgetConfig = Field(
        default_factory=TokenBudgetConfig,
        description="Token usage budget configuration",
    )

    # Framework settings
    plugin_paths: list[str] = Field(
        default_factory=list,
        description="Paths to search for plugins",
    )
    auto_reload: bool = Field(
        default=False,
        description="Whether to enable auto-reload in development",
    )

    @model_validator(mode="after")
    def load_from_config_file(self) -> "FastADKSettings":
        """Load configuration from a file if specified or from default paths."""
        # Get config path from environment or instance
        config_path = self.config_path

        # If no explicit config path, check default locations
        if not config_path:
            for path in DEFAULT_CONFIG_PATHS:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    config_path = expanded_path
                    break

        # Load config file if found
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)

        # Apply environment-specific overrides
        self._apply_environment_settings()

        return self

    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from a YAML or TOML file."""
        try:
            with open(config_path, encoding="utf-8") as f:
                if config_path.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f)
                elif config_path.endswith(".toml"):
                    import tomli

                    config_data = tomli.loads(f.read())
                else:
                    print(f"Unsupported config file format: {config_path}")
                    return

                # Update settings from file if it contains valid data
                if config_data and isinstance(config_data, dict):
                    self._update_from_dict(config_data)
                    print(f"Loaded configuration from {config_path}")
                else:
                    print(f"Warning: Empty or invalid configuration in {config_path}")
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {config_path}: {e}")
        except ImportError:
            print("tomli package not installed. Required for TOML config files.")
        except (OSError, PermissionError) as e:
            print(f"Error accessing configuration file {config_path}: {e}")
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Unexpected error loading configuration from {config_path}: {e}")

    def _update_from_dict(self, data: dict[str, Any]) -> None:
        """Update settings from a dictionary, handling nested objects."""
        for key, value in data.items():
            if hasattr(self, key):
                current_value = getattr(self, key)

                # Handle nested configuration objects
                if isinstance(current_value, BaseModel) and isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if hasattr(current_value, subkey):
                            setattr(current_value, subkey, subvalue)
                else:
                    setattr(self, key, value)

    def _apply_environment_settings(self) -> None:
        """Apply settings specific to the current environment."""
        # Adjust settings based on environment
        if self.environment == EnvironmentType.DEVELOPMENT:
            if self.telemetry.log_level == LogLevel.INFO:
                # Update in a way that works with Pydantic
                telemetry = self.telemetry.model_copy(deep=True)
                telemetry.log_level = LogLevel.DEBUG
                self.telemetry = telemetry

            self.auto_reload = True

        elif self.environment == EnvironmentType.PRODUCTION:
            # Production-specific settings
            telemetry = self.telemetry.model_copy(deep=True)
            telemetry.metrics_enabled = True
            self.telemetry = telemetry

            security = self.security.model_copy(deep=True)
            security.content_filtering = True
            self.security = security

        elif self.environment == EnvironmentType.TESTING:
            # Testing-specific settings
            telemetry = self.telemetry.model_copy(deep=True)
            telemetry.enabled = False
            self.telemetry = telemetry

            memory = self.memory.model_copy(deep=True)
            memory.backend_type = MemoryBackendType.IN_MEMORY
            self.memory = memory


# Create a global settings instance
settings = FastADKSettings()


def get_settings() -> FastADKSettings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> FastADKSettings:
    """Reload settings from environment and config files."""
    # Re-initialize settings
    new_settings = FastADKSettings()

    # Update the module-level settings
    # This approach doesn't require the global keyword
    globals()["settings"] = new_settings

    return new_settings
