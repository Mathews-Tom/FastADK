"""
Core agent module containing BaseAgent class and decorator implementations.

This module provides the foundation for agent creation in FastADK,
including the BaseAgent class and @Agent and @tool decorators.
"""
# pylint: disable=attribute-defined-outside-init, redefined-outer-name

import asyncio
import functools
import inspect
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .config import get_settings
from .exceptions import AgentError, ConfigurationError, ToolError

# Load environment variables from .env file
load_dotenv()

# Type definitions
T = TypeVar("T")
AgentMethod = Callable[..., Any]
ToolFunction = Callable[..., Any]

# Setup logging
logger = logging.getLogger("fastadk.agent")


class AgentMetadata(BaseModel):
    """A Pydantic model to store structured metadata about an agent."""

    name: str
    model: str
    description: str = ""
    system_prompt: str | None = None
    provider: str = "simulated"
    tools: list["ToolMetadata"] = Field(default_factory=list)


class ToolMetadata(BaseModel):
    """Metadata for a tool function."""

    name: str
    description: str
    function: Callable[..., Any]
    cache_ttl: int = 0  # Time-to-live for cached results in seconds
    timeout: int = 30  # Timeout in seconds
    retries: int = 0  # Number of retries on failure
    enabled: bool = True  # Whether the tool is enabled
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_type: type | None = None


class ProviderABC(ABC):
    """
    Abstract Base Class for LLM providers.

    This class defines the interface that all backend providers must implement.
    This allows FastADK to remain model-agnostic.
    """

    @abstractmethod
    async def initialize(self, metadata: AgentMetadata) -> Any:
        """
        Initializes the provider with the agent's metadata.

        This is where the provider would prepare the LLM, but the actual model
        instance might be lazy-loaded on the first run.

        Args:
            metadata: The agent's configuration.

        Returns:
            An internal representation of the agent instance for the provider.
        """

    @abstractmethod
    async def register_tool(
        self, agent_instance: Any, tool_metadata: ToolMetadata
    ) -> None:
        """
        Registers a tool's schema with the provider.

        Args:
            agent_instance: The provider's internal agent representation.
            tool_metadata: The metadata of the tool to register.
        """

    @abstractmethod
    async def run(self, agent_instance: Any, input_text: str, **kwargs: Any) -> str:
        """
        Executes the main agent logic with a given input.

        Args:
            agent_instance: The provider's internal agent representation.
            input_text: The user's prompt.
            **kwargs: Additional data, such as the `execute_tool` callback.

        Returns:
            The final, user-facing response from the LLM.
        """


class BaseAgent:
    """
    Base class for all FastADK agents.

    This class provides the core functionality for agent creation,
    tool management, and execution.
    """

    # Class variables for storing agent metadata
    _tools: ClassVar[dict[str, ToolMetadata]] = {}
    _model_name: ClassVar[str] = "gemini-1.5-pro"
    _description: ClassVar[str] = "A FastADK agent"
    _provider: ClassVar[str] = "gemini"

    def __init__(self) -> None:
        """Initialize the agent with configuration settings."""
        self.settings = get_settings()
        self.tools: dict[str, ToolMetadata] = {}
        self.tools_used: list[str] = []
        self.session_id: str | None = None
        self.memory_data: dict[str, Any] = {}

        # Initialize tools from class metadata
        self._initialize_tools()

        # Initialize model based on configuration
        self._initialize_model()

        logger.info(
            "Initialized agent %s with %d tools",
            self.__class__.__name__,
            len(self.tools),
        )

    def _initialize_tools(self) -> None:
        """Initialize tools from class metadata."""
        # Copy tools from class variable to instance
        self.tools = {}

        # Add any instance methods decorated as tools
        for name, method in inspect.getmembers(self, inspect.ismethod):
            # pylint: disable=protected-access
            if hasattr(method, "_is_tool") and method._is_tool:
                metadata = getattr(method, "_tool_metadata", {})
                self.tools[name] = ToolMetadata(
                    name=name,
                    description=metadata.get("description", method.__doc__ or ""),
                    function=method,
                    cache_ttl=metadata.get("cache_ttl", 0),
                    timeout=metadata.get("timeout", 30),
                    retries=metadata.get("retries", 0),
                    enabled=metadata.get("enabled", True),
                    parameters=metadata.get("parameters", {}),
                    return_type=metadata.get("return_type", None),
                )

    def _initialize_model(self) -> None:
        """Initialize the AI model based on configuration."""
        try:
            if self._provider == "gemini":
                self._initialize_gemini_model()
            elif self._provider == "openai":
                self._initialize_openai_model()
            elif self._provider == "anthropic":
                self._initialize_anthropic_model()
            elif self._provider == "simulated":
                # Use mock model for simulation and testing
                from fastadk.testing.utils import MockModel

                self.model = MockModel()  # type: ignore
                logger.info("Initialized simulated mock model")
            else:
                # If provider is unknown, try to use a mock model
                raise ConfigurationError(
                    f"Unsupported provider: {self._provider}. "
                    "Supported providers: 'gemini', 'openai', 'anthropic', 'simulated'"
                )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize model: {e}") from e

    def _initialize_gemini_model(self) -> None:
        """Initialize Gemini model."""
        # Using getattr to avoid FieldInfo errors in IDE
        api_key_var = getattr(self.settings.model, "api_key_env_var", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_var) or os.environ.get("GEMINI_API_KEY")

        if api_key:
            genai.configure(api_key=api_key)
            # Set default Gemini configuration
            self.model = genai.GenerativeModel(self._model_name)  # type: ignore
            logger.info("Initialized Gemini model %s", self._model_name)
        else:
            # For tests, use a mock model if no API key is available
            from fastadk.testing.utils import MockModel

            # pylint: disable=attribute-defined-outside-init
            self.model = MockModel()  # type: ignore
            logger.info(
                "Using mock model for %s (no API key available)",
                self._model_name,
            )

    def _initialize_openai_model(self) -> None:
        """Initialize OpenAI model."""
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY environment variable not set")

            # Initialize the client
            self.model = openai.OpenAI(api_key=api_key)  # type: ignore
            logger.info("Initialized OpenAI model %s", self._model_name)
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Install with: uv add openai"
            ) from exc

    def _initialize_anthropic_model(self) -> None:
        """Initialize Anthropic model."""
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "ANTHROPIC_API_KEY environment variable not set"
                )

            # Initialize the client
            self.model = anthropic.Anthropic(api_key=api_key)  # type: ignore
            logger.info("Initialized Anthropic model %s", self._model_name)
        except ImportError as exc:
            raise ImportError(
                "Anthropic package not installed. Install with: uv add anthropic"
            ) from exc

    async def run(self, user_input: str) -> str:
        """
        Run the agent with the given user input.

        This method processes the user input, potentially executes tools,
        and returns a response from the agent.

        Args:
            user_input: The user's input message

        Returns:
            The agent's response as a string
        """
        start_time = time.time()
        self.tools_used = []  # Reset tools used for this run

        try:
            # Simple implementation for now - just pass to model
            # In future versions, this will handle tool calling and memory
            response = await self._generate_response(user_input)

            # Log execution time
            execution_time = time.time() - start_time
            logger.info("Agent execution completed in %.2fs", execution_time)

            return response
        except Exception as e:
            logger.error("Error during agent execution: %s", e, exc_info=True)
            raise AgentError(f"Failed to process input: {e}") from e

    async def _generate_response(self, user_input: str) -> str:
        """Generate a response from the model."""
        try:
            # Handle different providers
            if self._provider == "gemini":
                return await self._generate_gemini_response(user_input)
            if self._provider == "openai":
                return await self._generate_openai_response(user_input)
            if self._provider == "anthropic":
                return await self._generate_anthropic_response(user_input)
            if self._provider == "simulated":
                # For simulated/mock model
                if hasattr(self.model, "generate_content"):
                    response = await asyncio.to_thread(
                        lambda: self.model.generate_content(user_input).text
                    )
                    return str(response)
                return f"Simulated response to: {user_input}"

            # If no provider matched
            raise AgentError(f"Unsupported provider: {self._provider}")
        except Exception as e:
            logger.error("Error generating response: %s", e, exc_info=True)
            raise AgentError(f"Failed to generate response: {e}") from e

    async def _generate_gemini_response(self, user_input: str) -> str:
        """Generate a response using the Gemini model."""
        response = await asyncio.to_thread(
            lambda: self.model.generate_content(user_input).text
        )
        return str(response)

    async def _generate_openai_response(self, user_input: str) -> str:
        """Generate a response using the OpenAI model."""
        # Handle both real OpenAI model and mock model
        if hasattr(self.model, "chat"):
            response = await asyncio.to_thread(
                lambda: self.model.chat.completions.create(  # type: ignore
                    model=self._model_name,
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=1000,
                )
            )
            return response.choices[0].message.content or ""  # type: ignore

        # Fallback for mock model
        return f"OpenAI mock response to: {user_input}"

    async def _generate_anthropic_response(self, user_input: str) -> str:
        """Generate a response using the Anthropic model."""
        # Handle both real Anthropic model and mock model
        if hasattr(self.model, "messages"):
            response = await asyncio.to_thread(
                lambda: self.model.messages.create(  # type: ignore
                    model=self._model_name,
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=1000,
                )
            )
            return response.content[0].text  # type: ignore

        # Fallback for mock model
        return f"Anthropic mock response to: {user_input}"

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if tool_name not in self.tools:
            raise ToolError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        if not tool.enabled:
            raise ToolError(f"Tool '{tool_name}' is disabled")

        self.tools_used.append(tool_name)
        logger.info("Executing tool '%s' with args: %s", tool_name, kwargs)

        # Execute with timeout and retry logic
        remaining_retries = tool.retries
        while True:
            try:
                # Execute tool with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: tool.function(**kwargs)),
                    timeout=tool.timeout,
                )
                return result

            except asyncio.TimeoutError:
                logger.warning("Tool '%s' timed out after %ds", tool_name, tool.timeout)
                if remaining_retries > 0:
                    remaining_retries -= 1
                    logger.info(
                        "Retrying tool '%s', %d retries left",
                        tool_name,
                        remaining_retries,
                    )
                    continue
                raise ToolError(
                    f"Tool '{tool_name}' timed out and max retries exceeded"
                ) from None

            except Exception as e:
                logger.error(
                    "Error executing tool '%s': %s", tool_name, e, exc_info=True
                )
                if remaining_retries > 0:
                    remaining_retries -= 1
                    logger.info(
                        "Retrying tool '%s', %d retries left",
                        tool_name,
                        remaining_retries,
                    )
                    continue
                raise ToolError(f"Tool '{tool_name}' failed: {e}") from e

    def on_start(self) -> None:
        """Hook called when the agent starts processing a request."""

    def on_finish(self, result: str) -> None:
        """Hook called when the agent finishes processing a request."""

    def on_error(self, error: Exception) -> None:
        """Hook called when the agent encounters an error."""


def Agent(
    model: str = "gemini-1.5-pro",
    description: str = "",
    provider: str = "gemini",
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator for creating FastADK agents.

    Args:
        model: The name of the model to use
        description: Description of the agent
        provider: The provider to use (gemini, etc.)
        **kwargs: Additional configuration options

    Returns:
        A decorator function that modifies the agent class
    """

    def decorator(cls: type[T]) -> type[T]:
        # Store metadata on the class
        # pylint: disable=protected-access
        cls._model_name = model  # type: ignore
        cls._description = description or cls.__doc__ or ""  # type: ignore
        cls._provider = provider  # type: ignore

        # Add any additional kwargs as class variables
        for key, value in kwargs.items():
            setattr(cls, f"_{key}", value)

        return cls

    return decorator


# pylint: disable=redefined-outer-name, redefined-builtin
def tool(
    cache_ttl: int = 0,
    timeout: int = 30,
    retries: int = 0,
    enabled: bool = True,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for tool functions that can be used by agents.

    Args:
        cache_ttl: Time-to-live for cached results in seconds
        timeout: Timeout in seconds
        retries: Number of retries on failure
        enabled: Whether the tool is enabled
        **kwargs: Additional metadata for the tool

    Returns:
        A decorator function that registers the tool
    """
    # Handle usage as @tool without parentheses
    if callable(cache_ttl):
        func = cache_ttl

        # Create a decorator with default values and apply it
        decorator_with_defaults = tool(cache_ttl=0, timeout=30, retries=0, enabled=True)
        return decorator_with_defaults(func)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get tool metadata from docstring and signature
        description = func.__doc__ or ""
        sig = inspect.signature(func)
        parameters = {}
        return_type = (
            sig.return_annotation
            if sig.return_annotation != inspect.Signature.empty
            else None
        )

        # Process parameters
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = (
                param.annotation if param.annotation != inspect.Signature.empty else Any
            )
            parameters[param_name] = {
                "type": param_type,
                "required": param.default == inspect.Parameter.empty,
            }

        # Create tool metadata
        tool_metadata = {
            "description": description,
            "cache_ttl": cache_ttl,
            "timeout": timeout,
            "retries": retries,
            "enabled": enabled,
            "parameters": parameters,
            "return_type": return_type,
        }
        tool_metadata.update(kwargs)

        # Store metadata on the function
        # pylint: disable=protected-access
        func._is_tool = True  # type: ignore
        func._tool_metadata = tool_metadata  # type: ignore

        # For standalone functions (not methods), register now
        if not any(param.name == "self" for param in sig.parameters.values()):
            # This is a standalone function, not a method
            # Register it with the global registry
            name = kwargs.get("name", func.__name__)
            BaseAgent._tools[name] = ToolMetadata(
                name=name,
                description=description,
                function=func,
                cache_ttl=cache_ttl,
                timeout=timeout,
                retries=retries,
                enabled=enabled,
                parameters=parameters,
                return_type=return_type,
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If this is the first time the method is called through the instance
            # make sure it's registered in the instance's tools dictionary
            if args and hasattr(args[0], "tools") and isinstance(args[0], BaseAgent):
                self_obj = args[0]
                method_name = func.__name__

                # Register the method in the instance's tools if not already there
                if method_name not in self_obj.tools:
                    self_obj.tools[method_name] = ToolMetadata(
                        name=method_name,
                        description=description,
                        function=getattr(self_obj, func.__name__),
                        cache_ttl=cache_ttl,
                        timeout=timeout,
                        retries=retries,
                        enabled=enabled,
                        parameters=parameters,
                        return_type=return_type,
                    )

            # Execute the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
