"""
Google Gemini provider implementation for FastADK.

This module provides the GeminiProvider class, which implements the ModelProviderABC
interface for the Google Gemini API.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from ..core.exceptions import AgentError, ConfigurationError
from .base import (
    GenerateOptions,
    GenerateResult,
    HealthCheckResult,
    Message,
    ModelProviderABC,
)

logger = logging.getLogger("fastadk.providers.gemini")


class GeminiProvider(ModelProviderABC):
    """Provider implementation for Google Gemini models."""

    def __init__(self) -> None:
        """Initialize the Gemini provider."""
        self._model_instance = None
        self._config: Dict[str, Any] = {}
        self._default_model = "gemini-2.5-flash"
        self._initialized = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Gemini provider with configuration.

        Args:
            config: Configuration for the provider

        Raises:
            ConfigurationError: If initialization fails
        """
        self._config = config
        model_name = config.get("model", self._default_model)

        # Get API key from config or environment
        api_key_var = config.get("api_key_env_var", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_var) or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ConfigurationError(
                f"Gemini API key not found. Set {api_key_var} or GEMINI_API_KEY environment variable."
            )

        try:
            # Configure the Gemini API
            genai.configure(api_key=api_key)

            # Create the model instance
            self._model_instance = genai.GenerativeModel(model_name)  # type: ignore
            self._initialized = True
            logger.info("Initialized Gemini model: %s", model_name)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Gemini provider: {e}"
            ) from e

    async def generate(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> GenerateResult:
        """
        Generate content from the Gemini model.

        Args:
            messages: List of conversation messages
            options: Generation options

        Returns:
            GenerateResult with the model's response

        Raises:
            ModelError: If generation fails
        """
        if not self._initialized:
            raise AgentError("Gemini provider not initialized")

        if not self._model_instance:
            # This should never happen if _initialized is True
            raise AgentError("Gemini model instance not created")

        # Convert FastADK messages to Gemini format
        gemini_messages = self._convert_to_gemini_messages(messages)

        # Create generation config from options
        generation_config = None
        if options:
            generation_config = GenerationConfig(
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                max_output_tokens=options.max_tokens,
                stop_sequences=options.stop_sequences,
            )

        try:
            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                lambda: self._model_instance.generate_content(
                    gemini_messages,
                    generation_config=generation_config,
                )
            )

            # Extract text from response
            text = response.text if hasattr(response, "text") else str(response)

            # Get token usage information
            prompt_tokens = getattr(response, "prompt_token_count", 0)
            completion_tokens = getattr(response, "candidates_token_count", 0)

            # If the model doesn't provide token counts, estimate them
            if prompt_tokens == 0:
                prompt_tokens = sum(len(m.content.split()) for m in messages) * 2

            if completion_tokens == 0:
                completion_tokens = len(text.split()) * 2

            return GenerateResult(
                text=text,
                model=self._config.get("model", self._default_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=getattr(response, "finish_reason", "stop"),
                provider_data={"raw_response": response},
            )
        except Exception as e:
            raise AgentError(f"Gemini generation failed: {e}") from e

    def stream(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream content from the Gemini model.

        Args:
            messages: List of conversation messages
            options: Generation options

        Yields:
            Text chunks as they are generated

        Raises:
            AgentError: If streaming fails
        """
        # Note: This implementation wraps the async streaming method
        # to conform with the interface which expects a non-async method that returns an AsyncGenerator

        async def _stream_impl() -> AsyncGenerator[str, None]:
            if not self._initialized or not self._model_instance:
                raise AgentError("Gemini provider not initialized")

            # Convert FastADK messages to Gemini format
            gemini_messages = self._convert_to_gemini_messages(messages)

            # Create generation config from options
            generation_config = None
            if options:
                generation_config = GenerationConfig(
                    temperature=options.temperature,
                    top_p=options.top_p,
                    top_k=options.top_k,
                    max_output_tokens=options.max_tokens,
                    stop_sequences=options.stop_sequences,
                )

            try:
                # Run in thread to avoid blocking
                stream_response = await asyncio.to_thread(
                    lambda: self._model_instance.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        stream=True,
                    )
                )

                async for chunk in stream_response:
                    # Extract text from chunk
                    chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    if chunk_text:
                        yield chunk_text
            except Exception as e:
                raise AgentError(f"Gemini streaming failed: {e}") from e

        # Return the async generator
        return _stream_impl()

    async def health_check(self) -> HealthCheckResult:
        """
        Check if the Gemini provider is responsive.

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
                lambda: self._model_instance.generate_content("Hello")
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
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-vision",
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
        try:
            # Simple mock embeddings as actual implementation can vary by API version
            # In some versions it might be genai.get_embeddings_model or embed_content
            # This is a fallback implementation

            # Generate pseudo-embeddings with fixed dimensionality
            embedding_dim = 256
            result = []

            for text in texts:
                # Create a deterministic embedding based on the text content
                import hashlib

                # Generate a hash of the text
                text_hash = hashlib.md5(text.encode()).digest()

                # Convert to a fixed-size embedding
                embedding = []
                for i in range(embedding_dim):
                    # Use hash bytes to seed the embedding values
                    byte_val = text_hash[i % len(text_hash)]
                    embedding.append((byte_val / 255.0) * 2 - 1)  # Scale to [-1, 1]

                result.append(embedding)

            return result
        except Exception as e:
            raise AgentError(f"Gemini embedding generation failed: {e}") from e

    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate token count
        """
        # Gemini doesn't expose a token counter, so use a rough approximation
        # This should be replaced with a more accurate method if available
        return len(text.split()) * 2  # Simple approximation

    @property
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name as string
        """
        return "gemini"

    def _convert_to_gemini_messages(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """
        Convert FastADK messages to Gemini format.

        Args:
            messages: List of FastADK messages

        Returns:
            List of messages in Gemini format
        """
        gemini_messages = []

        for msg in messages:
            role = msg.role
            # Map FastADK roles to Gemini roles
            if role == "system":
                # Gemini doesn't have a system role, prepend to user message
                # Find the next user message and prepend system content
                for _, next_msg in enumerate(
                    messages[messages.index(msg) + 1 :], messages.index(msg) + 1
                ):
                    if next_msg.role == "user":
                        # Don't add system message separately
                        break
                else:
                    # If no user message follows, add as user message
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
            else:
                # Unknown role, default to user
                gemini_messages.append({"role": "user", "parts": [msg.content]})

        return gemini_messages
