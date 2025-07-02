"""
Core agent classes, decorators, and providers for the FastADK framework.

This module provides the primary building blocks for creating AI agents:
- **@Agent:** A class decorator that transforms a Python class into a fully functional AI agent.
- **@tool:** A function decorator to register methods as tools available to an agent.
- **BaseAgent:** A base class that agents can inherit from, providing lifecycle hooks.
- **ProviderABC:** An abstract base class for creating new backend providers (e.g., for different LLMs).
- **GeminiProvider:** A concrete implementation for Google's Gemini models.
- **SimulatedProvider:** A mock provider for testing and development.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, TypeVar, cast, get_type_hints

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastadk.core.exceptions import AgentError, ConfigurationError

logger = logging.getLogger("fastadk")

# --- Pydantic Models for Metadata ---


class ToolMetadata(BaseModel):
    """A Pydantic model to store structured metadata about a tool."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_type: str = "any"


class AgentMetadata(BaseModel):
    """A Pydantic model to store structured metadata about an agent."""

    name: str
    model: str
    description: str = ""
    system_prompt: str | None = None
    provider: str = "simulated"
    tools: list[ToolMetadata] = Field(default_factory=list)


# --- Provider Abstraction ---


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


# --- Concrete Provider Implementations ---


class SimulatedProvider(ProviderABC):
    """A mock provider for development and testing that does not make real API calls."""

    async def initialize(self, m: AgentMetadata) -> dict[str, Any]:
        """Initializes a simulated agent instance."""
        return {"metadata": m.model_dump(), "tools": []}

    async def register_tool(self, i: Any, t: ToolMetadata) -> None:
        """Registers a tool with the simulated agent."""
        i["tools"].append(t.model_dump())

    async def run(self, agent_instance: Any, input_text: str, **kwargs: Any) -> str:
        """Returns a simple, predictable response for testing."""
        return f"Simulated response for: '{input_text}'"


class GeminiProvider(ProviderABC):
    """An LLM provider for Google's Gemini family of models."""

    # This mapping is encapsulated within the provider that needs it.
    _PYTHON_TO_GEMINI_TYPE = {
        "str": "STRING",
        "int": "INTEGER",
        "float": "NUMBER",
        "bool": "BOOLEAN",
        "list": "ARRAY",
        "dict": "OBJECT",
    }

    def __init__(self) -> None:
        """
        Initializes the Gemini provider by loading the API key and configuring the library.

        Raises:
            ConfigurationError: If the Gemini library is not installed or the API key is missing.
        """
        try:
            import google.generativeai as genai

            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "GEMINI_API_KEY not found in environment or .env file."
                )
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ConfigurationError(
                "The 'google-generativeai' and 'python-dotenv' packages are required "
                "to use the GeminiProvider. Please install them."
            )

    async def initialize(self, metadata: AgentMetadata) -> dict[str, Any]:
        """Prepares the agent instance but lazy-loads the actual model."""
        return {"metadata": metadata.model_dump(), "model": None, "tools": []}

    async def register_tool(
        self, agent_instance: dict[str, Any], tool_metadata: ToolMetadata
    ) -> None:
        """Converts FastADK tool metadata into the format required by the Gemini API."""
        properties, required = {}, []
        for name, info in tool_metadata.parameters.items():
            py_type = info.get("type", "str")
            gemini_type = self._PYTHON_TO_GEMINI_TYPE.get(py_type, "STRING")
            properties[name] = {"type": gemini_type, "description": info["description"]}
            if info.get("required"):
                required.append(name)

        declaration = self.genai.types.FunctionDeclaration(
            name=tool_metadata.name,
            description=tool_metadata.description,
            parameters={
                "type": "OBJECT",
                "properties": properties,
                "required": required,
            },
        )
        agent_instance["tools"].append(declaration)

    async def run(
        self, agent_instance: dict[str, Any], input_text: str, **kwargs: Any
    ) -> str:
        """
        Runs the agent, handling the full logic cycle of prompting, tool calling,
        and generating a final response.
        """
        # Lazy-load the model on the first run to ensure all tools are registered.
        if agent_instance.get("model") is None:
            model_name = agent_instance["metadata"]["model"]
            system_instruction = agent_instance["metadata"].get("system_prompt")
            agent_tools = self.genai.types.Tool(
                function_declarations=agent_instance["tools"]
            )
            agent_instance["model"] = self.genai.GenerativeModel(
                model_name, tools=[agent_tools], system_instruction=system_instruction
            )

        model = agent_instance["model"]
        chat = model.start_chat()
        response = await asyncio.to_thread(chat.send_message, input_text)

        try:
            part = response.candidates[0].content.parts[0]
            if not part.function_call:
                return str(response.text)

            # --- Tool Calling Logic ---
            fc = part.function_call
            logger.info(f"Model requesting tool: {fc.name}({dict(fc.args)})")

            execute_tool = kwargs.get("execute_tool")
            if not execute_tool:
                raise AgentError(
                    "execute_tool callback was not provided to the provider."
                )

            result = await execute_tool(fc.name, **dict(fc.args))

            # Serialize the tool's result to a string for the next prompt.
            try:
                tool_response_content = json.dumps(result, indent=2)
            except TypeError:
                tool_response_content = str(result)

            # Create a new prompt with the tool's output to get the final answer.
            second_prompt = (
                f"The tool '{fc.name}' was called and returned the following result:\n\n"
                f"```json\n{tool_response_content}\n```\n\n"
                "Based on this information, please provide a final, user-friendly answer."
            )

            final_response = await asyncio.to_thread(chat.send_message, second_prompt)
            return str(final_response.text)

        except Exception as e:
            logger.error("Error during agent run: %s", e, exc_info=True)
            return f"An error occurred: {e}"


PROVIDER_REGISTRY = {"simulated": SimulatedProvider, "gemini": GeminiProvider}

# --- Core Decorators ---


T = TypeVar("T", bound=Callable[..., Any])


def tool(
    _func: T | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Callable[[T], T] | T:
    """
    A decorator to register a class method as a tool available to an agent.

    It automatically infers the tool's name, description, and parameter schema
    from the function's signature and docstring, reducing boilerplate.

    Args:
        name: An optional override for the tool's name. Defaults to the function name.
        description: An optional override for the tool's description. Defaults to the
                        first line of the function's docstring.
    """

    def decorator(func: T) -> T:
        tool_name = name or func.__name__
        doc = (func.__doc__ or "").strip()
        tool_desc = doc.split("\n")[0]
        params, type_hints = {}, get_type_hints(func)

        # Infer parameter schema from type hints and docstrings.
        for p_name, p in inspect.signature(func).parameters.items():
            if p_name == "self":
                continue
            params[p_name] = {
                "description": description,
                "type": getattr(type_hints.get(p_name, Any), "__name__", "any").lower(),
                "required": p.default is inspect.Parameter.empty,
            }

        for line in doc.split("\n"):
            if line.strip().startswith((":param", "@param")):
                try:
                    _, p_name, desc = line.split(":", 2)
                    p_name = p_name.strip()
                    if p_name in params:
                        params[p_name]["description"] = desc.strip()
                except ValueError:
                    pass

        meta = ToolMetadata(
            name=tool_name, description=tool_desc, parameters=params, **kwargs
        )

        # The wrapper must be async to handle both sync and async tool functions.
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Apply the tool metadata to the wrapper function
        async_wrapper._tool_metadata = meta
        return cast(T, async_wrapper)

    return decorator if _func is None else decorator(_func)


def Agent(
    *,
    name: str | None = None,
    model: str,
    description: str | None = None,
    system_prompt: str | Path | None = None,
    provider: str = "simulated",
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    A class decorator that transforms a Python class into a fully functional AI agent.

    It handles initialization, tool registration, and the execution lifecycle.

    Args:
        name: An optional name for the agent. Defaults to the class name.
        model: The identifier for the LLM to be used (e.g., 'gemini-1.5-pro').
        description: A short description of the agent's purpose.
        system_prompt: A detailed system prompt, either as a string or a Path object
                        pointing to a text file.
        provider: The name of the backend provider to use (e.g., 'gemini').
    """

    def decorator(cls: type) -> type:
        prompt_text = None
        if isinstance(system_prompt, Path):
            try:
                prompt_text = system_prompt.read_text(encoding="utf-8")
            except Exception as e:
                raise ConfigurationError(f"Error reading system prompt file: {e}")
        elif isinstance(system_prompt, str):
            prompt_text = system_prompt

        metadata = AgentMetadata(
            name=name or cls.__name__,
            model=model,
            description=description or (cls.__doc__ or "").strip(),
            system_prompt=prompt_text,
            provider=provider,
        )
        # Set class attributes
        cls._agent_metadata = metadata  # type: ignore

        # Find all tool methods on the class
        tools = {}
        for attr in dir(cls):
            obj = getattr(cls, attr)
            if hasattr(obj, "_tool_metadata"):
                tools[attr] = obj._tool_metadata

        # Store tools using a more type-safe approach
        # Manually adding an attribute to the class
        object.__setattr__(cls, "_tools", tools)

        original_init = cls.__init__

        @wraps(original_init)
        def init_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            # Simply call the original __init__ directly - skip the complex logic
            # to avoid mypy issues with instance.__init__ access
            # We use a type ignore comment specifically for the misc category that catches this issue
            original_init(self, *args, **kwargs)  # type: ignore[misc]

            # Set up provider and instance attributes
            self.provider = PROVIDER_REGISTRY[provider]()
            self._initialized = False
            self._metadata = metadata

            # Get tools - we use getattr with default to avoid mypy errors
            tools_dict = getattr(cls, "_tools", {})
            self._tool_metadata = list(tools_dict.values())

        async def execute_tool(self: Any, tool_name: str, **kwargs: Any) -> Any:
            """Executes a tool method on the agent instance by its name."""
            method = getattr(self, tool_name)
            return await method(**kwargs)

        async def run(self: Any, input_text: str, **kwargs: Any) -> str:
            """
            Initializes the agent if needed, then runs the main execution logic.
            This method is attached to the decorated class, replacing the placeholder
            in BaseAgent.
            """
            if not self._initialized:
                self.agent_instance = await self.provider.initialize(self._metadata)
                for meta in self._tool_metadata:
                    await self.provider.register_tool(self.agent_instance, meta)
                self._initialized = True

            kwargs["execute_tool"] = self.execute_tool
            result = await self.provider.run(self.agent_instance, input_text, **kwargs)
            return str(result)

        # Attach the implemented methods to the decorated class
        # Use object.__setattr__ for better type safety with mypy
        object.__setattr__(cls, "__init__", init_wrapper)
        object.__setattr__(cls, "run", run)
        object.__setattr__(cls, "execute_tool", execute_tool)
        return cls

    return decorator


class BaseAgent:
    """
    A base class for creating FastADK agents.

    While the @Agent decorator handles most of the implementation, inheriting
    from BaseAgent provides type hinting, standard lifecycle hooks, and a clear
    structure for your agent classes.
    """

    _agent_metadata: ClassVar[AgentMetadata]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """The __init__ method is preserved for your own initialization logic."""

    async def run(self, input_text: str, **kwargs: Any) -> str:
        """This method is dynamically implemented by the @Agent decorator."""
        raise NotImplementedError

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """This method is dynamically implemented by the @Agent decorator."""
        raise NotImplementedError

    # --- Lifecycle Hooks ---
    async def on_start(self, input_text: str, **kwargs: Any) -> None:
        """A hook that runs at the beginning of the agent's execution cycle."""

    async def on_finish(self, response: str, **kwargs: Any) -> None:
        """A hook that runs after the agent has generated a final response."""

    async def on_error(self, error: Exception, **kwargs: Any) -> None:
        """A hook that runs if an exception occurs during the agent's execution."""
