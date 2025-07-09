"""
OpenAI provider implementation for FastADK.

This module provides the OpenAIProvider class, which implements the ModelProviderABC
interface for the OpenAI API.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from openai.types.chat import ChatCompletionMessageParam

from ..core.exceptions import AgentError, ConfigurationError
from .base import (
    GenerateOptions,
    GenerateResult,
    HealthCheckResult,
    Message,
    ModelProviderABC,
)

logger = logging.getLogger("fastadk.providers.openai")


class OpenAIProvider(ModelProviderABC):
    """Provider implementation for OpenAI models."""

    def __init__(self) -> None:
        """Initialize the OpenAI provider."""
        self._client = None
        self._config: Dict[str, Any] = {}
        self._default_model = "gpt-4.1"
        self._initialized = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the OpenAI provider with configuration.

        Args:
            config: Configuration for the provider

        Raises:
            ConfigurationError: If initialization fails
        """
        self._config = config
        model_name = config.get("model", self._default_model)

        # Get API key from config or environment
        api_key_var = config.get("api_key_env_var", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_var) or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ConfigurationError(
                f"OpenAI API key not found. Set {api_key_var} or OPENAI_API_KEY environment variable."
            )

        try:
            # Configure the OpenAI client
            self._client = openai.OpenAI(api_key=api_key)
            self._initialized = True
            logger.info("Initialized OpenAI model: %s", model_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize OpenAI provider: {e}"
            ) from e

    async def generate(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> GenerateResult:
        """
        Generate content from the OpenAI model.

        Args:
            messages: List of conversation messages
            options: Generation options

        Returns:
            GenerateResult with the model's response

        Raises:
            ModelError: If generation fails
        """
        if not self._initialized:
            raise AgentError("OpenAI provider not initialized")

        if not self._client:
            # This should never happen if _initialized is True
            raise AgentError("OpenAI client not created")

        # Convert FastADK messages to OpenAI format
        openai_messages = self._convert_to_openai_messages(messages)

        # Get model name
        model_name = self._config.get("model", self._default_model)

        # Create generation parameters from options
        params = {
            "model": model_name,
            "messages": openai_messages,
        }

        if options:
            if options.temperature is not None:
                params["temperature"] = options.temperature
            if options.max_tokens is not None:
                params["max_tokens"] = options.max_tokens
            if options.top_p is not None:
                params["top_p"] = options.top_p
            if options.stop_sequences is not None:
                params["stop"] = options.stop_sequences

        try:
            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                lambda: self._client.chat.completions.create(**params)  # type: ignore
            )

            # Extract text from response
            text = response.choices[0].message.content or ""

            # Get token usage information
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            return GenerateResult(
                text=text,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=response.choices[0].finish_reason,
                provider_data={"raw_response": response},
            )
        except Exception as e:
            raise AgentError(f"OpenAI generation failed: {e}") from e

    def stream(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream content from the OpenAI model.

        Args:
            messages: List of conversation messages
            options: Generation options

        Yields:
            Text chunks as they are generated

        Raises:
            AgentError: If streaming fails
        """

        async def _stream_impl() -> AsyncGenerator[str, None]:
            if not self._initialized or not self._client:
                raise AgentError("OpenAI provider not initialized")

            # Convert FastADK messages to OpenAI format
            openai_messages = self._convert_to_openai_messages(messages)

            # Get model name
            model_name = self._config.get("model", self._default_model)

            # Create generation parameters from options
            params = {
                "model": model_name,
                "messages": openai_messages,
                "stream": True,
            }

            if options:
                if options.temperature is not None:
                    params["temperature"] = options.temperature
                if options.max_tokens is not None:
                    params["max_tokens"] = options.max_tokens
                if options.top_p is not None:
                    params["top_p"] = options.top_p
                if options.stop_sequences is not None:
                    params["stop"] = options.stop_sequences

            try:
                # Run in thread to avoid blocking
                stream_response = await asyncio.to_thread(
                    lambda: self._client.chat.completions.create(**params)  # type: ignore
                )

                # Process streaming response
                async for chunk in stream_response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
            except Exception as e:
                raise AgentError(f"OpenAI streaming failed: {e}") from e

        # Return the async generator
        return _stream_impl()

    async def health_check(self) -> HealthCheckResult:
        """
        Check if the OpenAI provider is responsive.

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
            await asyncio.to_thread(
                lambda: self._client.chat.completions.create(
                    model=self._config.get("model", self._default_model),
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
            )
            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status="ok",
                latency_ms=latency_ms,
                details={
                    "model": self._config.get("model", self._default_model),
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
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4.1",
            "gpt-3.5-turbo",
        ]

    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ModelError: If embedding generation fails
        """
        if not self._initialized or not self._client:
            raise AgentError("OpenAI provider not initialized")

        try:
            # Get embeddings using OpenAI's embeddings API
            embedding_model = "text-embedding-3-small"
            response = await asyncio.to_thread(
                lambda: self._client.embeddings.create(
                    model=embedding_model,
                    input=texts,
                )
            )

            # Extract and return embeddings
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise AgentError(f"OpenAI embedding generation failed: {e}") from e

    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate token count
        """
        try:
            # Try to use tiktoken if available
            import tiktoken

            model = self._config.get("model", self._default_model)
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except (ImportError, Exception):
            # Fall back to simple approximation if tiktoken not available
            return len(text.split()) * 1.3  # Simple approximation

    @property
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name as string
        """
        return "openai"

    def _convert_to_openai_messages(
        self, messages: List[Message]
    ) -> List[ChatCompletionMessageParam]:
        """
        Convert FastADK messages to OpenAI format.

        Args:
            messages: List of FastADK messages

        Returns:
            List of messages in OpenAI format
        """
        openai_messages: List[ChatCompletionMessageParam] = []

        for msg in messages:
            role = msg.role
            # Map FastADK roles to OpenAI roles
            if role not in ["system", "user", "assistant"]:
                # Default unknown roles to user
                role = "user"

            openai_messages.append({"role": role, "content": msg.content})

        return openai_messages
