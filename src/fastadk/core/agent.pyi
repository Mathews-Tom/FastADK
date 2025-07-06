"""
Type stubs for the agent module.

This file provides precise type annotations for the agent module, enabling
IDE autocompletion and static type checking with mypy.
"""

# pylint: disable=unused-argument, unnecessary-ellipsis

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, overload

from pydantic import BaseModel

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type[Any])
R = TypeVar("R")  # Return type for tools

# Core types
AgentMethod = Callable[..., Any]
ToolFunction = Callable[..., Any]
AsyncToolFunction = Callable[..., asyncio.Future[Any]]

class ToolMetadata(BaseModel, Generic[R]):
    """Metadata for a tool function with generic return type."""

    name: str
    description: str
    function: Callable[..., R]
    cache_ttl: int
    timeout: int
    retries: int
    enabled: bool
    parameters: dict[str, Any]
    return_type: type[R] | None

class AgentMetadata(BaseModel):
    """A Pydantic model to store structured metadata about an agent."""

    name: str
    model: str
    description: str
    system_prompt: str | None
    provider: str
    tools: list[ToolMetadata[Any]]

class ProviderABC(ABC):
    """Abstract Base Class for LLM providers."""

    @abstractmethod
    async def initialize(self, metadata: AgentMetadata) -> Any: ...
    @abstractmethod
    async def register_tool(
        self, agent_instance: Any, tool_metadata: ToolMetadata[Any]
    ) -> None: ...
    @abstractmethod
    async def run(self, agent_instance: Any, input_text: str, **kwargs: Any) -> str: ...

class BaseAgent:
    """Base class for all FastADK agents."""

    # Class variables
    _tools: ClassVar[dict[str, ToolMetadata[Any]]]
    _model_name: ClassVar[str]
    _description: ClassVar[str]
    _provider: ClassVar[str]
    _system_message: ClassVar[str | None]

    # Instance variables
    settings: Any
    tools: dict[str, ToolMetadata[Any]]
    tools_used: list[str]
    session_id: str | None
    memory_data: dict[str, Any]
    model: Any
    litellm_mode: str
    litellm_endpoint: str
    last_response: str

    def __init__(self) -> None: ...
    def _initialize_tools(self) -> None: ...
    def _initialize_model(self) -> None: ...
    def _initialize_gemini_model(self) -> None: ...
    def _initialize_openai_model(self) -> None: ...
    def _initialize_anthropic_model(self) -> None: ...
    def _initialize_litellm_model(self) -> None: ...
    async def run(self, user_input: str) -> str:
        """Run the agent with the given user input.

        Args:
            user_input: The user's input message

        Returns:
            The agent's response as a string
        """
        ...

    async def _generate_response(self, user_input: str) -> str:
        """Generate a response from the model.

        Args:
            user_input: The user's input message

        Returns:
            The generated response
        """
        ...

    async def _generate_gemini_response(self, user_input: str) -> str:
        """Generate a response using the Gemini model.

        Args:
            user_input: The user's input message

        Returns:
            The generated response from Gemini
        """
        ...

    async def _generate_openai_response(self, user_input: str) -> str:
        """Generate a response using the OpenAI model.

        Args:
            user_input: The user's input message

        Returns:
            The generated response from OpenAI
        """
        ...

    async def _generate_anthropic_response(self, user_input: str) -> str:
        """Generate a response using the Anthropic model.

        Args:
            user_input: The user's input message

        Returns:
            The generated response from Anthropic
        """
        ...

    async def _generate_litellm_response(self, user_input: str) -> str:
        """Generate a response using the LiteLLM client.

        Args:
            user_input: The user's input message

        Returns:
            The generated response from LiteLLM
        """
        ...

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        ...

    def on_start(self) -> None:
        """Hook called when the agent starts processing a request."""
        ...

    def on_finish(self, result: str) -> None:
        """Hook called when the agent finishes processing a request."""
        ...

    def on_error(self, error: Exception) -> None:
        """Hook called when the agent encounters an error."""
        ...

# Agent decorator with overloads
@overload
def Agent(
    model: str = ..., description: str = ..., provider: str = ..., **kwargs: Any
) -> Callable[[type[C]], type[C]]: ...

# Tool decorator with overloads - variant 1: @tool
@overload
def tool(func: F) -> F: ...

# Tool decorator with overloads - variant 2: @tool(...)
@overload
def tool(
    *,
    cache_ttl: int = ...,
    timeout: int = ...,
    retries: int = ...,
    enabled: bool = ...,
    **kwargs: Any,
) -> Callable[[F], F]: ...
