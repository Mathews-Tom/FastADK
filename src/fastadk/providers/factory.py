"""
Provider factory module for FastADK.

This module implements the factory pattern for creating model providers,
making it easy to switch between different LLM backends.
"""

import logging
from typing import Any, Dict, Optional, Type

from ..core.exceptions import ConfigurationError
from .base import ModelProviderABC, ModelStub

logger = logging.getLogger("fastadk.provider_factory")


class ProviderFactory:
    """
    Factory class for creating model providers.

    This class manages the creation of model providers based on configuration,
    implementing the factory pattern to decouple provider implementation details
    from the rest of the system.
    """

    _providers: Dict[str, Type[ModelProviderABC]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: Type[ModelProviderABC]
    ) -> None:
        """
        Register a provider class with the factory.

        Args:
            name: The name to register the provider under
            provider_class: The provider class to register
        """
        cls._providers[name] = provider_class
        logger.debug("Registered provider: %s", name)

    @classmethod
    def create(
        cls, provider_name: str, config: Optional[Dict[str, Any]] = None
    ) -> ModelProviderABC:
        """
        Create a new provider instance.

        Args:
            provider_name: The name of the provider to create
            config: Optional configuration for the provider

        Returns:
            An initialized provider instance

        Raises:
            ConfigurationError: If the provider is not registered or initialization fails
        """
        if not config:
            config = {}

        # Handle the case where provider is specified but not registered
        if provider_name not in cls._providers:
            # Try to dynamically import the provider
            try:
                cls._try_import_provider(provider_name)
            except ImportError as e:
                if provider_name == "simulated" or provider_name == "stub":
                    # Return a stub provider for testing
                    logger.info("Using stub provider for '%s'", provider_name)
                    return ModelStub(name=provider_name)

                # If not a test provider, raise an error
                raise ConfigurationError(
                    f"Provider '{provider_name}' is not registered and could not be imported: {e}"
                ) from e

        # Get the provider class and create an instance
        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            raise ConfigurationError(f"Provider '{provider_name}' is not registered")

        try:
            provider = provider_class()
            # Initialize will be called asynchronously later
            return provider
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def _try_import_provider(cls, provider_name: str) -> None:
        """
        Try to dynamically import a provider module.

        Args:
            provider_name: The name of the provider to import

        Raises:
            ImportError: If the provider module cannot be imported
        """
        try:
            if provider_name == "gemini":
                import google.generativeai  # noqa: F401

                from ..providers.gemini import GeminiProvider

                cls.register_provider("gemini", GeminiProvider)
            elif provider_name == "openai":
                import openai  # noqa: F401

                from ..providers.openai import OpenAIProvider

                cls.register_provider("openai", OpenAIProvider)
            elif provider_name == "anthropic":
                import anthropic  # noqa: F401

                from ..providers.anthropic import AnthropicProvider

                cls.register_provider("anthropic", AnthropicProvider)
            elif provider_name == "litellm":
                import litellm  # noqa: F401

                from ..providers.litellm import LiteLLMProvider

                cls.register_provider("litellm", LiteLLMProvider)
            else:
                # Try looking for a plugin provider
                try:
                    # This would be extended with proper plugin discovery
                    plugin_module = __import__(f"fastadk_provider_{provider_name}")
                    provider_class = getattr(
                        plugin_module, f"{provider_name.capitalize()}Provider"
                    )
                    cls.register_provider(provider_name, provider_class)
                except (ImportError, AttributeError) as e:
                    raise ImportError(
                        f"Could not find provider plugin for '{provider_name}': {e}"
                    ) from e
        except ImportError as e:
            raise ImportError(
                f"Required package for provider '{provider_name}' is not installed: {e}"
            ) from e
