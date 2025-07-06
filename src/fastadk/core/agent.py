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
from typing import Any, ClassVar, Dict, Optional, TypeVar

import google.generativeai as genai
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ..tokens.models import TokenBudget
from ..tokens.utils import extract_token_usage_from_response, track_token_usage
from .config import get_settings
from .exceptions import (
    AgentError,
    ConfigurationError,
    ExceptionTracker,
    OperationTimeoutError,
    ToolError,
)

# Load environment variables from .env file
load_dotenv()

# Dictionary to store registered agent classes
_registered_agents: dict[str, type["BaseAgent"]] = {}


def get_registered_agent(name: str) -> type["BaseAgent"] | None:
    """
    Get a registered agent class by name.

    Args:
        name: Name of the agent class

    Returns:
        Agent class if found, None otherwise
    """
    return _registered_agents.get(name)


def register_agent(agent_class: type["BaseAgent"]) -> None:
    """
    Register an agent class.

    Args:
        agent_class: The agent class to register
    """
    name = agent_class.__name__
    _registered_agents[name] = agent_class
    logging.debug("Registered agent class: %s", name)


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
    _system_message: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Initialize the agent with configuration settings."""
        self.settings = get_settings()
        self.tools: dict[str, ToolMetadata] = {}
        self.tools_used: list[str] = []
        self.session_id: str | None = None
        self.memory_data: dict[str, Any] = {}
        self.litellm_mode: str = "sdk"
        self.litellm_endpoint: str = "http://localhost:8000"
        self.last_response: str = ""

        # Initialize token budget if tracking is enabled
        self.token_budget: Optional[TokenBudget] = None
        # Access actual attributes on the settings objects, not the Field definitions
        if getattr(self.settings.model, "track_tokens", False):
            token_budget_settings = self.settings.token_budget
            self.token_budget = TokenBudget(
                max_tokens_per_request=getattr(
                    token_budget_settings, "max_tokens_per_request", None
                ),
                max_tokens_per_session=getattr(
                    token_budget_settings, "max_tokens_per_session", None
                ),
                max_cost_per_request=getattr(
                    token_budget_settings, "max_cost_per_request", None
                ),
                max_cost_per_session=getattr(
                    token_budget_settings, "max_cost_per_session", None
                ),
                warn_at_percent=getattr(token_budget_settings, "warn_at_percent", 80.0),
            )

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
            elif self._provider == "litellm":
                self._initialize_litellm_model()
            elif self._provider == "simulated":
                # Use mock model for simulation and testing
                from fastadk.testing.utils import MockModel

                self.model = MockModel()  # type: ignore
                logger.info("Initialized simulated mock model")
            else:
                # If provider is unknown, try to use a mock model
                raise ConfigurationError(
                    f"Unsupported provider: {self._provider}. "
                    "Supported providers: 'gemini', 'openai', 'anthropic', 'litellm', 'simulated'"
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
            # Check cache for this prompt if caching is enabled
            cache_response = await self._check_cache(user_input)
            if cache_response:
                logger.info("Using cached response for input")
                self.last_response = cache_response
                return cache_response

            # Handle different providers
            if self._provider == "gemini":
                response_text = await self._generate_gemini_response(user_input)
            elif self._provider == "openai":
                response_text = await self._generate_openai_response(user_input)
            elif self._provider == "anthropic":
                response_text = await self._generate_anthropic_response(user_input)
            elif self._provider == "litellm":
                response_text = await self._generate_litellm_response(user_input)
            elif self._provider == "simulated":
                # For simulated/mock model
                if hasattr(self.model, "generate_content"):
                    response = await asyncio.to_thread(
                        lambda: self.model.generate_content(user_input).text
                    )
                    response_text = str(response)
                else:
                    response_text = f"Simulated response to: {user_input}"

                # Generate simulated token usage for demonstration purposes
                if getattr(self.settings.model, "track_tokens", False):
                    # Create simulated token usage based on input length
                    prompt_tokens = len(user_input.split())
                    completion_tokens = len(response_text.split())

                    # Create a simulated response object with usage data
                    from ..tokens.models import TokenUsage

                    usage = TokenUsage(
                        prompt_tokens=prompt_tokens
                        * 2,  # Simulate token encoding (more tokens than words)
                        completion_tokens=completion_tokens * 2,
                        model=self._model_name,
                        provider="simulated",
                    )

                    # Get custom price if available
                    custom_price = getattr(
                        self.settings.model, "custom_price_per_1k", {}
                    )
                    track_token_usage(usage, self.token_budget, custom_price)
            else:
                # If no provider matched
                raise AgentError(f"Unsupported provider: {self._provider}")

            # Cache the response
            await self._cache_response(user_input, response_text)

            # Store the response for potential tool skipping logic
            self.last_response = response_text
            return response_text
        except Exception as e:
            logger.error("Error generating response: %s", e, exc_info=True)
            raise AgentError(f"Failed to generate response: {e}") from e

    async def _check_cache(self, user_input: str) -> str | None:
        """
        Check if we have a cached response for this input.

        Args:
            user_input: The user's input message

        Returns:
            Cached response if available, None otherwise
        """
        try:
            from .cache import default_cache_manager

            # Only use cache if enabled for this model
            cache_ttl = getattr(self.settings.model, "response_cache_ttl", 0)
            if cache_ttl <= 0:
                return None

            # Create a cache key from the model, provider, and input
            cache_key = {
                "model": self._model_name,
                "provider": self._provider,
                "input": user_input,
            }

            # Try to get from cache
            cached_response = await default_cache_manager.get(cache_key)
            return cached_response
        except Exception as e:
            # Log but don't fail if cache check fails
            logger.warning(f"Error checking cache: {e}")
            return None

    async def _cache_response(self, user_input: str, response: str) -> None:
        """
        Cache a response for future use.

        Args:
            user_input: The user's input message
            response: The model's response
        """
        try:
            from .cache import default_cache_manager

            # Only cache if enabled for this model
            cache_ttl = getattr(self.settings.model, "response_cache_ttl", 0)
            if cache_ttl <= 0:
                return

            # Create a cache key from the model, provider, and input
            cache_key = {
                "model": self._model_name,
                "provider": self._provider,
                "input": user_input,
            }

            # Cache the response
            await default_cache_manager.set(cache_key, response, ttl=cache_ttl)
        except Exception as e:
            # Log but don't fail if caching fails
            logger.warning(f"Error caching response: {e}")

    async def _generate_gemini_response(self, user_input: str) -> str:
        """Generate a response using the Gemini model."""
        response = await asyncio.to_thread(
            lambda: self.model.generate_content(user_input)
        )

        # Extract the response text
        response_text = response.text if hasattr(response, "text") else str(response)

        # Track token usage if enabled
        if getattr(self.settings.model, "track_tokens", False):
            usage = extract_token_usage_from_response(
                response, "gemini", self._model_name
            )
            if usage:
                # Get custom price if available
                custom_price = getattr(self.settings.model, "custom_price_per_1k", {})
                track_token_usage(usage, self.token_budget, custom_price)

        return response_text

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

            # Track token usage if enabled
            if getattr(self.settings.model, "track_tokens", False):
                usage = extract_token_usage_from_response(
                    response, "openai", self._model_name
                )
                if usage:
                    # Get custom price if available
                    custom_price = getattr(
                        self.settings.model, "custom_price_per_1k", {}
                    )
                    track_token_usage(
                        usage,
                        self.token_budget,
                        custom_price,
                    )

                    # Check if we exceeded budget limits
                    if (
                        self.token_budget
                        and usage
                        and self.token_budget.max_tokens_per_request
                        and usage.total_tokens
                        > self.token_budget.max_tokens_per_request
                    ):
                        logger.warning(
                            "Token limit exceeded: %d > %d",
                            usage.total_tokens,
                            self.token_budget.max_tokens_per_request,
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

            # Track token usage if enabled
            if getattr(self.settings.model, "track_tokens", False):
                usage = extract_token_usage_from_response(
                    response, "anthropic", self._model_name
                )
                if usage:
                    # Get custom price if available
                    custom_price = getattr(
                        self.settings.model, "custom_price_per_1k", {}
                    )
                    track_token_usage(
                        usage,
                        self.token_budget,
                        custom_price,
                    )

            return response.content[0].text  # type: ignore

        # Fallback for mock model
        return f"Anthropic mock response to: {user_input}"

    def _initialize_litellm_model(self) -> None:
        """Initialize LiteLLM model."""
        try:
            # Get API key and configuration
            api_key_var = getattr(
                self.settings.model, "api_key_env_var", "LITELLM_API_KEY"
            )
            api_key = os.environ.get(api_key_var) or os.environ.get("LITELLM_API_KEY")

            # Get mode configuration (sdk or proxy)
            mode = getattr(self.settings.model, "litellm_mode", "sdk")

            if mode == "proxy":
                # Configure for proxy mode
                endpoint = getattr(
                    self.settings.model, "litellm_endpoint", "http://localhost:8000"
                )
                litellm.api_base = endpoint

                if api_key:
                    litellm.api_key = api_key

                # Store mode and endpoint for later use
                self.litellm_mode = "proxy"
                self.litellm_endpoint = endpoint
                logger.info(
                    "Initialized LiteLLM in proxy mode with endpoint %s", endpoint
                )
            else:
                # SDK mode is the default
                if api_key:
                    litellm.api_key = api_key

                self.litellm_mode = "sdk"
                logger.info("Initialized LiteLLM in SDK mode")

            # Store LiteLLM client instance
            self.model = litellm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LiteLLM package not installed. Install with: uv add litellm"
            ) from exc

    async def _generate_litellm_response(self, user_input: str) -> str:
        """Generate a response using the LiteLLM client."""
        # Create the messages array for LiteLLM (similar to OpenAI format)
        messages = [{"role": "user", "content": user_input}]

        # Add system message if available
        system_message = getattr(self, "_system_message", None)
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        try:
            # Call LiteLLM completion
            response = await asyncio.to_thread(
                lambda: litellm.completion(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=1000,
                )
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Track token usage if enabled
            if getattr(self.settings.model, "track_tokens", False):
                usage = extract_token_usage_from_response(
                    response, "litellm", self._model_name
                )
                if usage:
                    # Get custom price if available
                    custom_price = getattr(
                        self.settings.model, "custom_price_per_1k", {}
                    )
                    track_token_usage(usage, self.token_budget, custom_price)

            return response_text or ""
        except Exception as e:  # noqa: BLE001
            logger.error("Error generating response with LiteLLM: %s", str(e))
            # Fallback for error cases
            return f"Error generating response: {str(e)}"

    async def execute_tool(
        self,
        tool_name: str,
        skip_if_response_contains: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute
            skip_if_response_contains: List of strings that, if found in the LLM response,
                                        indicate the tool call can be skipped
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution or a message indicating the tool was skipped
        """
        if tool_name not in self.tools:
            raise ToolError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        if not tool.enabled:
            raise ToolError(f"Tool '{tool_name}' is disabled")

        # Check if we should skip the tool execution based on LLM response
        if skip_if_response_contains and hasattr(self, "last_response"):
            last_response = getattr(self, "last_response", "")
            for phrase in skip_if_response_contains:
                if phrase.lower() in last_response.lower():
                    logger.info(
                        "Skipping tool '%s' execution because response already contains relevant info",
                        tool_name,
                    )
                    return {
                        "skipped": True,
                        "reason": f"Response already contains '{phrase}'",
                        "tool_name": tool_name,
                    }

        # If not skipped, proceed with execution
        self.tools_used.append(tool_name)
        logger.info("Executing tool '%s' with args: %s", tool_name, kwargs)

        # Check if tool function is a coroutine
        is_coroutine = asyncio.iscoroutinefunction(tool.function)

        # Execute with timeout and retry logic
        remaining_retries = tool.retries
        while True:
            try:
                # Execute tool with timeout
                if is_coroutine:
                    # If it's already a coroutine, just await it
                    result = await asyncio.wait_for(
                        tool.function(**kwargs),
                        timeout=tool.timeout,
                    )
                else:
                    # Run sync function in a thread pool
                    result = await asyncio.wait_for(
                        asyncio.to_thread(lambda: tool.function(**kwargs)),
                        timeout=tool.timeout,
                    )

                # Try to cache the result if caching is enabled
                from .cache import default_cache_manager

                cache_ttl = getattr(tool, "cache_ttl", 0)
                if cache_ttl > 0:
                    # Create a cache key from the tool name and arguments
                    cache_key = {
                        "tool": tool_name,
                        "args": kwargs,
                    }
                    await default_cache_manager.set(cache_key, result, ttl=cache_ttl)

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
                timeout_error = OperationTimeoutError(
                    message=f"Tool '{tool_name}' timed out and max retries exceeded",
                    error_code="TOOL_TIMEOUT",
                    details={
                        "tool_name": tool_name,
                        "timeout_seconds": tool.timeout,
                        "retries_attempted": tool.retries,
                    },
                )
                ExceptionTracker.track_exception(timeout_error)
                raise timeout_error from None

            except Exception as e:
                logger.error(
                    "Error executing tool '%s': %s", tool_name, str(e), exc_info=True
                )
                if remaining_retries > 0:
                    remaining_retries -= 1
                    logger.info(
                        "Retrying tool '%s', %d retries left",
                        tool_name,
                        remaining_retries,
                    )
                    continue
                tool_error = ToolError(
                    message=f"Tool '{tool_name}' failed: {e}",
                    error_code="TOOL_EXECUTION_ERROR",
                    details={
                        "tool_name": tool_name,
                        "original_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                ExceptionTracker.track_exception(tool_error)
                raise tool_error from e

    def on_start(self) -> None:
        """Hook called when the agent starts processing a request."""

    def on_finish(self, result: str) -> None:
        """Hook called when the agent finishes processing a request."""

    def on_error(self, error: Exception) -> None:
        """Hook called when the agent encounters an error."""

    def reset_token_budget(self) -> None:
        """Reset the token budget session counters."""
        if self.token_budget:
            self.token_budget.reset_session()
            logger.info("Token budget session counters reset")

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get the current token usage statistics."""
        if self.token_budget:
            return {
                "session_tokens_used": self.token_budget.session_tokens_used,
                "session_cost": self.token_budget.session_cost,
                "has_request_limit": self.token_budget.max_tokens_per_request
                is not None,
                "has_session_limit": self.token_budget.max_tokens_per_session
                is not None,
                "has_cost_limit": (
                    self.token_budget.max_cost_per_request is not None
                    or self.token_budget.max_cost_per_session is not None
                ),
            }
        return {"token_tracking_enabled": False}


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

        # Register the agent class
        if issubclass(
            cls, BaseAgent
        ):  # Make sure we only register BaseAgent subclasses
            register_agent(cls)  # type: ignore

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
