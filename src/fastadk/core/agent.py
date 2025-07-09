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
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field

from ..providers.base import GenerateOptions, Message
from ..providers.factory import ProviderFactory
from ..tokens.models import TokenBudget
from ..tokens.utils import track_token_usage
from .config import get_settings
from .exceptions import AgentError, ConfigurationError
from .plugin import AgentPlugin, Plugin, get_supported_plugins
from .plugin_manager import default_plugin_manager
from .tool_manager import ToolManager

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
    _model_name: ClassVar[str] = "gemini-2.5-pro"
    _description: ClassVar[str] = "A FastADK agent"
    _provider: ClassVar[str] = "gemini"
    _system_message: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Initialize the agent with configuration settings."""
        self.settings = get_settings()
        self.session_id: str | None = None
        self.memory_data: dict[str, Any] = {}
        self.last_response: str = ""
        self.tools_used: List[str] = []  # For backward compatibility

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

        # Create tool manager and initialize tools
        self.tool_manager = ToolManager(self)
        self._initialize_tools()

        # For backward compatibility
        self.tools = self.tool_manager.tools

        # Initialize model provider
        self._initialize_provider()

        # Initialize plugins
        self.plugin_manager = default_plugin_manager
        self._active_plugins: Dict[str, Plugin] = {}
        self._initialize_plugins()

        logger.info(
            "Initialized agent %s with %d tools",
            self.__class__.__name__,
            len(self.tool_manager.tools),
        )

        # Emit agent initialized event - ensure event loop is running or handle gracefully
        try:
            # Get running event loop and create task
            asyncio.get_running_loop()
            asyncio.create_task(
                self.plugin_manager.emit_event(
                    "agent:initialized",
                    {"agent": self, "agent_class": self.__class__.__name__},
                )
            )
        except RuntimeError:
            # No running event loop, log this but don't fail initialization
            logger.debug(
                "No running event loop for agent initialization event. This is normal in test environments."
            )

    def _initialize_tools(self) -> None:
        """Initialize tools from class metadata and instance methods."""
        # First, register any class-level tools
        if self._tools:
            self.tool_manager.register_tools(self._tools)

        # Add any instance methods decorated as tools
        for name, method in inspect.getmembers(self, inspect.ismethod):
            # pylint: disable=protected-access
            if hasattr(method, "_is_tool") and method._is_tool:
                metadata = getattr(method, "_tool_metadata", {})
                self.tool_manager.register_tool(
                    ToolMetadata(
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
                )

    def _initialize_plugins(self) -> None:
        """Initialize plugins for this agent."""
        # Check if the class supports plugins
        supported_plugins = get_supported_plugins(self.__class__)

        for plugin_class in supported_plugins:
            try:
                # Create plugin instance
                plugin = plugin_class()

                # Register the plugin
                asyncio.create_task(
                    self.plugin_manager.register_plugin_instance(plugin)
                )

                # Store for direct access
                self._active_plugins[plugin.name] = plugin

                # For agent-specific plugins, call the enhancement method
                if isinstance(plugin, AgentPlugin):
                    asyncio.create_task(plugin.on_agent_initialized(self))

                logger.info(
                    "Initialized plugin %s for agent %s",
                    plugin.name,
                    self.__class__.__name__,
                )
            except Exception as e:
                logger.error(
                    "Error initializing plugin %s: %s", plugin_class.__name__, str(e)
                )

    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register a plugin with this agent.

        Args:
            plugin: The plugin to register
        """
        if plugin.name in self._active_plugins:
            logger.warning(
                "Plugin %s already registered with agent %s",
                plugin.name,
                self.__class__.__name__,
            )
            return

        # Register with plugin manager
        asyncio.create_task(self.plugin_manager.register_plugin_instance(plugin))

        # Store for direct access
        self._active_plugins[plugin.name] = plugin

        # For agent-specific plugins, call the enhancement method
        if isinstance(plugin, AgentPlugin):
            asyncio.create_task(plugin.on_agent_initialized(self))

        logger.info(
            "Registered plugin %s with agent %s", plugin.name, self.__class__.__name__
        )

    def _initialize_provider(self) -> None:
        """Initialize the model provider based on configuration."""
        try:
            # Create provider configuration
            provider_config = {
                "model": self._model_name,
                "system_message": self._system_message,
            }

            # Create the provider instance using the factory
            self.provider = ProviderFactory.create(self._provider, provider_config)

            # Initialize the provider (will be done asynchronously on first run)
            logger.info(
                "Provider %s registered for model %s", self._provider, self._model_name
            )
        except Exception as exc:
            logger.error("Failed to initialize provider: %s", str(exc), exc_info=True)
            raise ConfigurationError(f"Failed to initialize provider: {exc}") from exc

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
        self.tool_manager.reset()  # Reset tools used for this run

        # Emit agent:run_started event
        await self.plugin_manager.emit_event(
            "agent:run_started",
            {
                "agent": self,
                "agent_class": self.__class__.__name__,
                "input": user_input,
                "timestamp": start_time,
            },
        )

        try:
            # Call the on_start hook
            self.on_start()

            # Initialize the provider if needed
            await self._ensure_provider_initialized()

            # Generate response from the model
            response = await self._generate_response(user_input)

            # Call the on_finish hook
            self.on_finish(response)

            # Log execution time
            execution_time = time.time() - start_time
            logger.info("Agent execution completed in %.2fs", execution_time)

            # Emit agent:run_completed event
            await self.plugin_manager.emit_event(
                "agent:run_completed",
                {
                    "agent": self,
                    "agent_class": self.__class__.__name__,
                    "input": user_input,
                    "response": response,
                    "duration": execution_time,
                    "success": True,
                    "timestamp": time.time(),
                },
            )

            return response
        except Exception as exc:
            # Call the on_error hook
            self.on_error(exc)

            execution_time = time.time() - start_time

            # Emit agent:run_error event
            await self.plugin_manager.emit_event(
                "agent:run_error",
                {
                    "agent": self,
                    "agent_class": self.__class__.__name__,
                    "input": user_input,
                    "error": str(exc),
                    "duration": execution_time,
                    "timestamp": time.time(),
                },
            )

            logger.error("Error during agent execution: %s", str(exc), exc_info=True)
            raise AgentError(f"Failed to process input: {exc}") from exc

    async def _ensure_provider_initialized(self) -> None:
        """Ensure the provider is initialized."""
        try:
            # Create a configuration dict for the provider
            config = {
                "model": self._model_name,
                "system_message": self._system_message,
            }

            # Add any model-specific configuration from settings
            if hasattr(self.settings, "model"):
                model_settings = getattr(self.settings, "model", {})
                for key, value in model_settings.__dict__.items():
                    if key.startswith("_"):
                        continue
                    config[key] = value

            # Initialize the provider
            await self.provider.initialize(config)
        except Exception as exc:
            logger.error("Failed to initialize provider: %s", str(exc), exc_info=True)
            raise ConfigurationError(f"Failed to initialize provider: {exc}") from exc

    async def _generate_response(self, user_input: str) -> str:
        """Generate a response from the model."""
        try:
            # Check cache for this prompt if caching is enabled
            cache_response = await self._check_cache(user_input)
            if cache_response:
                logger.info("Using cached response for input")
                self.last_response = cache_response
                return cache_response

            # Prepare messages for the provider
            messages = self._prepare_messages(user_input)

            # Set generation options
            options = GenerateOptions(
                temperature=getattr(self.settings.model, "temperature", 0.7),
                max_tokens=getattr(self.settings.model, "max_tokens", 1000),
                top_p=getattr(self.settings.model, "top_p", None),
                top_k=getattr(self.settings.model, "top_k", None),
                stop_sequences=getattr(self.settings.model, "stop_sequences", None),
            )

            # Emit LLM request event
            start_time = time.time()
            await self.plugin_manager.emit_event(
                "llm:request",
                {
                    "agent": self,
                    "agent_class": self.__class__.__name__,
                    "model": self._model_name,
                    "provider": self._provider,
                    "temperature": options.temperature,
                    "max_tokens": options.max_tokens,
                    "messages": [msg.dict() for msg in messages],
                    "timestamp": start_time,
                },
            )

            # Generate response
            result = await self.provider.generate(messages, options)
            response_text = result.text

            # Emit LLM response event
            duration = time.time() - start_time
            await self.plugin_manager.emit_event(
                "llm:response",
                {
                    "agent": self,
                    "agent_class": self.__class__.__name__,
                    "model": self._model_name,
                    "provider": self._provider,
                    "tokens": result.prompt_tokens + result.completion_tokens,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "duration": duration,
                    "timestamp": time.time(),
                },
            )

            # Track token usage if enabled
            if (
                getattr(self.settings.model, "track_tokens", False)
                and self.token_budget
            ):
                custom_price = getattr(self.settings.model, "custom_price_per_1k", {})
                # Convert GenerateResult to TokenUsage for tracking
                from ..tokens.models import TokenUsage

                token_usage = TokenUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    model=self._model_name,
                    provider=self._provider,
                )

                track_token_usage(token_usage, self.token_budget, custom_price)

            # Cache the response
            await self._cache_response(user_input, response_text)

            # Store the response for potential tool skipping logic
            self.last_response = response_text
            return response_text
        except Exception as exc:
            logger.error("Error generating response: %s", str(exc), exc_info=True)
            raise AgentError(f"Failed to generate response: {exc}") from exc

    def _prepare_messages(self, user_input: str) -> List[Message]:
        """
        Prepare messages for the provider.

        Args:
            user_input: The user's input message

        Returns:
            List of messages ready for the provider
        """
        messages = []

        # Add system message if available
        if self._system_message:
            messages.append(Message(role="system", content=self._system_message))

        # Add user message
        messages.append(Message(role="user", content=user_input))

        return messages

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
        except Exception as exc:
            # Log but don't fail if cache check fails
            logger.warning("Error checking cache: %s", str(exc))
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
        except Exception as exc:
            # Log but don't fail if caching fails
            logger.warning("Error caching response: %s", str(exc))

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
        # Emit tool called event
        start_time = time.time()
        await self.plugin_manager.emit_event(
            "tool:called",
            {
                "agent": self,
                "agent_class": self.__class__.__name__,
                "tool_name": tool_name,
                "args": kwargs,
                "timestamp": start_time,
            },
        )

        success = True
        result = None

        try:
            # Delegate to the tool manager
            result = await self.tool_manager.execute_tool(
                tool_name, skip_if_response_contains, **kwargs
            )

            # For backward compatibility
            if (
                tool_name not in self.tools_used
                and tool_name in self.tool_manager.tools_used
            ):
                self.tools_used.append(tool_name)
        except Exception as e:
            success = False
            # Re-raise the exception
            raise e
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Emit tool completed event
            await self.plugin_manager.emit_event(
                "tool:completed",
                {
                    "agent": self,
                    "agent_class": self.__class__.__name__,
                    "tool_name": tool_name,
                    "duration": duration,
                    "success": success,
                    "timestamp": time.time(),
                },
            )

        return result

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
    model: str = "gemini-2.5-pro",
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
            if (
                args
                and hasattr(args[0], "tool_manager")
                and isinstance(args[0], BaseAgent)
            ):
                self_obj = args[0]
                method_name = func.__name__

                # Register the method in the instance's tools if not already there
                if method_name not in self_obj.tool_manager.tools:
                    self_obj.tool_manager.register_tool(
                        ToolMetadata(
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
                    )

            # Execute the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
