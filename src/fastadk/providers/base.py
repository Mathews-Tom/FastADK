"""
Base classes and interfaces for model providers in FastADK.

This module defines the abstract interfaces that all model providers must implement,
allowing FastADK to maintain a consistent API across different LLM providers.
"""

import abc
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation with a model."""

    role: str = Field(
        ...,
        description="The role of the message sender (e.g., 'user', 'assistant', 'system')",
    )
    content: str = Field(..., description="The content of the message")

    class Config:
        """Configuration for the Message model."""

        extra = "allow"  # Allow extra fields for provider-specific extensions


class HealthCheckResult(BaseModel):
    """Result of a health check on a model provider."""

    status: str = Field(
        ..., description="Status of the provider ('ok', 'degraded', 'error')"
    )
    latency_ms: float = Field(..., description="Response time in milliseconds")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class GenerateOptions(BaseModel):
    """Options for generating content from a model."""

    temperature: Optional[float] = Field(
        None, description="Sampling temperature, affecting randomness"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(
        None, description="Sequences that will stop generation when encountered"
    )

    class Config:
        """Configuration for the GenerateOptions model."""

        extra = "allow"  # Allow extra fields for provider-specific options


class GenerateResult(BaseModel):
    """Result of a content generation request."""

    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        ..., description="Number of tokens in the completion"
    )
    total_tokens: int = Field(..., description="Total tokens used")
    finish_reason: Optional[str] = Field(
        None, description="Reason why generation finished (e.g., 'stop', 'length')"
    )
    provider_data: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific response data"
    )


class ModelProviderABC(abc.ABC):
    """
    Abstract Base Class for LLM providers.

    This class defines the interface that all model providers must implement
    to be compatible with FastADK.
    """

    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the provider with configuration settings.

        Args:
            config: Configuration settings specific to this provider

        Raises:
            ConfigurationError: If initialization fails due to configuration issues
        """

    @abc.abstractmethod
    async def generate(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> GenerateResult:
        """
        Generate content from the model based on a list of messages.

        Args:
            messages: List of messages in the conversation
            options: Optional parameters to control generation behavior

        Returns:
            GenerateResult containing the model's response and metadata

        Raises:
            ModelError: If the model fails to generate content
            ConfigurationError: If the request is invalid
        """

    @abc.abstractmethod
    def stream(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream content from the model, yielding chunks as they're generated.

        Args:
            messages: List of messages in the conversation
            options: Optional parameters to control generation behavior

        Yields:
            Text chunks as they are generated

        Raises:
            ModelError: If the model fails to generate content
            ConfigurationError: If the request is invalid
        """

    @abc.abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Check if the model provider is responsive and healthy.

        Returns:
            HealthCheckResult with status and diagnostics
        """

    @abc.abstractmethod
    def supported_models(self) -> List[str]:
        """
        Get a list of models supported by this provider.

        Returns:
            List of model identifiers that can be used with this provider
        """

    @abc.abstractmethod
    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors, one for each input text

        Raises:
            ModelError: If embedding generation fails
        """

    @abc.abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name as a string
        """


class ModelStub(ModelProviderABC):
    """
    A stub implementation of ModelProviderABC for testing.

    This class provides a minimal working implementation that can be used
    for testing or as a template for new provider implementations.
    """

    def __init__(self, name: str = "stub"):
        """
        Initialize the stub provider.

        Args:
            name: Name for this stub provider
        """
        self._name = name
        self._initialized = False
        self._models = ["stub-model-small", "stub-model-large"]

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the stub provider."""
        self._initialized = True
        if config.get("models"):
            self._models = config["models"]

    async def generate(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> GenerateResult:
        """Generate a simple response."""
        last_message = messages[-1].content if messages else ""
        return GenerateResult(
            text=f"Stub response to: {last_message}",
            model=self._models[0],
            prompt_tokens=len(last_message.split()),
            completion_tokens=10,
            total_tokens=len(last_message.split()) + 10,
            finish_reason="stop",
        )

    async def stream(
        self, messages: List[Message], options: Optional[GenerateOptions] = None
    ) -> AsyncGenerator[str, None]:
        """Stream a simple response."""
        response = f"Stub response to: {messages[-1].content if messages else ''}"
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.1)

    async def health_check(self) -> HealthCheckResult:
        """Perform a mock health check."""
        return HealthCheckResult(
            status="ok", latency_ms=10.0, details={"initialized": self._initialized}
        )

    def supported_models(self) -> List[str]:
        """Return list of supported models."""
        return self._models

    async def embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        # Return a simple mock embedding for each text
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_token_count(self, text: str) -> int:
        """Count tokens in a simple way."""
        # Simple approximation: words + punctuation
        return len(text.split())

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._name
