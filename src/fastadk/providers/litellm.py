"""
LiteLLM provider implementation for FastADK.

This module provides the LiteLLMProvider class, which implements the ModelProviderABC
interface for LiteLLM, allowing access to multiple LLM providers through a single interface.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import litellm

    # In newer versions of litellm, there is an async_completion utility
    # If not available, we'll create a wrapper around the synchronous version
    try:
        from litellm.utils import async_completion
    except ImportError:
        async_completion = None
except ImportError:
    litellm = None
    async_completion = None

from ..core.exceptions import AgentError, ConfigurationError
from .base import (
    GenerateOptions,
    GenerateResult,
    HealthCheckResult,
    Message,
    ModelProviderABC,
)

# Use standard logging until we confirm observability module is available
try:
    from ..observability.logger import get_logger

    logger = get_logger("fastadk.providers.litellm")
except ImportError:
    logger = logging.getLogger("fastadk.providers.litellm")


class LiteLLMProvider(ModelProviderABC):
    """Provider implementation for LiteLLM, which supports multiple LLM providers."""

    def __init__(self) -> None:
        """Initialize the LiteLLM provider."""
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._default_model = "gpt-3.5-turbo"
        self._mode = "sdk"  # "sdk" or "proxy"
        self._endpoint = "http://localhost:8000"
        self._model_name = self._default_model

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LiteLLM provider with configuration.

        Args:
            config: Configuration for the provider

        Raises:
            ConfigurationError: If initialization fails
        """
        if litellm is None:
            raise ConfigurationError(
                "LiteLLM package not installed. Install with: uv add litellm"
            )

        self._config = config
        self._model_name = config.get("model", self._default_model)

        # Get API key from config or environment
        api_key_var = config.get("api_key_env_var", "LITELLM_API_KEY")
        api_key = os.environ.get(api_key_var) or os.environ.get("LITELLM_API_KEY")

        # Get mode configuration (sdk or proxy)
        self._mode = config.get("litellm_mode", "sdk")

        try:
            # Configure LiteLLM
            if self._mode == "proxy":
                # Configure for proxy mode
                self._endpoint = config.get("litellm_endpoint", "http://localhost:8000")
                litellm.api_base = self._endpoint

                if api_key:
                    litellm.api_key = api_key

                logger.info(
                    "Initialized LiteLLM in proxy mode with endpoint: %s",
                    {"endpoint": self._endpoint},
                )
            else:
                # SDK mode is the default
                if api_key:
                    litellm.api_key = api_key

                logger.info("Initialized LiteLLM in SDK mode")

            self._initialized = True
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize LiteLLM provider: {e}"
            ) from e

    async def generate(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> GenerateResult:
        """
        Generate content from the LiteLLM.

        Args:
            messages: List of conversation messages
            options: Generation options

        Returns:
            GenerateResult with the model's response

        Raises:
            AgentError: If generation fails
        """
        if not self._initialized:
            raise AgentError("LiteLLM provider not initialized")

        # Convert FastADK messages to LiteLLM/OpenAI format
        litellm_messages = self._convert_to_litellm_messages(messages)

        try:
            # Set options
            completion_options = {
                "model": self._config.get("model", self._default_model),
                "messages": litellm_messages,
            }

            # Add optional parameters
            if options:
                if options.temperature is not None:
                    completion_options["temperature"] = options.temperature
                if options.max_tokens is not None:
                    completion_options["max_tokens"] = options.max_tokens
                if options.top_p is not None:
                    completion_options["top_p"] = options.top_p
                if options.stop_sequences:
                    completion_options["stop"] = options.stop_sequences

            # Call LiteLLM completion using the async implementation if available,
            # otherwise create a wrapper around the synchronous version
            logger.debug(
                "Generating content with LiteLLM",
                extra={"model": completion_options["model"]},
            )
            if async_completion is not None:
                response = await async_completion(**completion_options)
            else:
                # Create an async wrapper around the synchronous implementation
                response = await asyncio.to_thread(
                    litellm.completion, **completion_options
                )

            # Extract text from response
            response_text = (
                response.choices[0].message.content if response.choices else ""
            )

            # Get token usage information
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

            return GenerateResult(
                text=response_text,
                model=self._config.get("model", self._default_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=(
                    getattr(response.choices[0], "finish_reason", "stop")
                    if response.choices
                    else "stop"
                ),
                provider_data={"raw_response": response},
            )
        except Exception as e:
            raise AgentError(f"LiteLLM generation failed: {e}") from e

    async def stream_content(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """
        Internal async implementation of streaming content from LiteLLM.

        Args:
            messages: List of conversation messages
            options: Generation options

        Yields:
            Text chunks as they are generated

        Raises:
            AgentError: If streaming fails
        """
        if not self._initialized:
            raise AgentError("LiteLLM provider not initialized")

        # Convert FastADK messages to LiteLLM/OpenAI format
        litellm_messages = self._convert_to_litellm_messages(messages)

        try:
            # Set options
            completion_options = {
                "model": self._config.get("model", self._default_model),
                "messages": litellm_messages,
                "stream": True,
            }

            # Add optional parameters
            if options:
                if options.temperature is not None:
                    completion_options["temperature"] = options.temperature
                if options.max_tokens is not None:
                    completion_options["max_tokens"] = options.max_tokens
                if options.top_p is not None:
                    completion_options["top_p"] = options.top_p
                if options.stop_sequences:
                    completion_options["stop"] = options.stop_sequences

            logger.debug(
                "Streaming content with LiteLLM",
                extra={"model": completion_options["model"]},
            )

            # Use async implementation if available, otherwise create a wrapper
            if async_completion is not None:
                stream_response = await async_completion(**completion_options)
            else:
                # Create an async wrapper around the synchronous implementation
                stream_response = await asyncio.to_thread(
                    litellm.completion, **completion_options
                )

            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except (ValueError, TypeError) as e:
            # Handle specific exceptions with detailed error messages
            raise AgentError(f"LiteLLM streaming failed due to input error: {e}") from e
        except Exception as e:
            # Handle general exceptions
            raise AgentError(f"LiteLLM streaming failed: {e}") from e

    def stream(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream content from LiteLLM.

        Args:
            messages: List of conversation messages
            options: Generation options

        Yields:
            Text chunks as they are generated

        Raises:
            AgentError: If streaming fails
        """
        # This method serves as a bridge to maintain the interface while using
        # the async implementation
        return self.stream_content(messages, options)

    async def health_check(self) -> HealthCheckResult:
        """
        Check if the LiteLLM provider is responsive.

        Returns:
            HealthCheckResult with status information
        """
        if not self._initialized:
            return HealthCheckResult(
                status="error",
                latency_ms=0.0,
                details={"error": "Provider not initialized"},
            )

        try:
            # Simple health check - generate a short response
            start_time = time.time()

            test_completion = {
                "model": self._config.get("model", self._default_model),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }

            logger.debug(
                "Performing health check", extra={"model": test_completion["model"]}
            )
            # Use async implementation if available, otherwise create a wrapper
            if async_completion is not None:
                await async_completion(**test_completion)
            else:
                # Create an async wrapper around the synchronous implementation
                await asyncio.to_thread(litellm.completion, **test_completion)
            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status="ok",
                latency_ms=latency_ms,
                details={
                    "model": self._config.get("model", self._default_model),
                    "mode": self._mode,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                status="error",
                latency_ms=0.0,
                details={"error": str(e)},
            )

    def supported_models(self) -> List[str]:
        """
        Get a list of models supported by this provider.

        Returns:
            List of supported model names
        """
        # LiteLLM supports many models, this is just a subset
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "claude-instant",
            "claude-2",
            "gemini-pro",
            "llama-2-70b",
            "mistral-medium",
        ]

    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            AgentError: If embedding generation fails
            NotImplementedError: If embedding generation is not supported
        """
        try:
            if not hasattr(litellm, "embedding"):
                raise NotImplementedError(
                    "Embedding functionality not available in this version of LiteLLM"
                )

            # Use litellm's embedding function if available
            embedding_options = {
                "model": self._config.get("embedding_model", "text-embedding-ada-002"),
                "input": texts,
            }

            logger.debug("Generating embeddings", extra={"count": len(texts)})

            # This should be replaced with async_embedding when available in litellm
            result = []
            for text in texts:
                # Use async implementation if available, otherwise create a wrapper
                if async_completion is not None:
                    embedding_response = await async_completion(
                        model=embedding_options["model"],
                        messages=[{"role": "user", "content": text}],
                        return_embeddings=True,
                    )
                else:
                    # Create an async wrapper around the synchronous implementation
                    embedding_response = await asyncio.to_thread(
                        litellm.completion,
                        model=embedding_options["model"],
                        messages=[{"role": "user", "content": text}],
                        return_embeddings=True,
                    )
                result.append(embedding_response.get("embeddings", []))

            return result
        except NotImplementedError:
            # Re-raise the NotImplementedError without modification
            raise
        except AgentError:
            # Re-raise AgentError without modification
            raise
        except (ValueError, TypeError) as e:
            # Convert specific exceptions to AgentError with context
            raise AgentError(f"LiteLLM embedding error with input data: {e}") from e
        except Exception as e:
            # Convert other exceptions to AgentError with context
            raise AgentError(f"LiteLLM embedding generation failed: {e}") from e

    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate token count
        """
        if not self._initialized or not hasattr(litellm, "token_counter"):
            # Simple approximation if LiteLLM is not available
            return int(len(text.split()) * 1.5)

        try:
            # LiteLLM has a function for token counting
            return litellm.token_counter(
                model=self._config.get("model", self._default_model), text=text
            )
        except (ValueError, TypeError) as e:
            # Log specific errors with appropriate context
            logger.warning(
                "Token counting failed due to input error",
                extra={
                    "error": str(e),
                    "model": self._config.get("model", self._default_model),
                },
            )
            return int(len(text.split()) * 1.5)
        except Exception as e:
            # Log general errors
            logger.warning(
                "Token counting failed, using approximation",
                extra={
                    "error": str(e),
                    "model": self._config.get("model", self._default_model),
                },
            )
            # Simple fallback approximation
            return int(len(text.split()) * 1.5)

    @property
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name as string
        """
        return "litellm"

    def _convert_to_litellm_messages(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """
        Convert FastADK messages to LiteLLM/OpenAI format.

        Args:
            messages: List of FastADK messages

        Returns:
            List of messages in LiteLLM/OpenAI format
        """
        litellm_messages = []

        for msg in messages:
            litellm_messages.append({"role": msg.role, "content": msg.content})

        return litellm_messages


# Register this provider with the factory
try:
    from .factory import ProviderFactory

    ProviderFactory.register_provider("litellm", LiteLLMProvider)
except ImportError:
    # Factory not available, likely during initialization or in tests
    pass
