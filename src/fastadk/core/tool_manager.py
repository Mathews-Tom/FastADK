"""
Tool management module for FastADK agents.

This module provides the ToolManager class, which handles registration,
execution, and caching of tools for agents.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .cache import default_cache_manager
from .exceptions import ExceptionTracker, OperationTimeoutError, ToolError

if TYPE_CHECKING:
    from .agent import BaseAgent, ToolMetadata

logger = logging.getLogger("fastadk.tool_manager")


class ToolManager:
    """
    Manages tools for a FastADK agent.

    This class handles tool registration, execution, and caching,
    implementing the Single Responsibility Principle by extracting
    tool-related functionality from the BaseAgent class.
    """

    def __init__(self, agent: "BaseAgent") -> None:
        """
        Initialize the ToolManager.

        Args:
            agent: The agent this manager is associated with
        """
        self.agent = agent
        self.tools: Dict[str, "ToolMetadata"] = {}
        self.tools_used: List[str] = []

    def register_tool(self, tool_metadata: "ToolMetadata") -> None:
        """
        Register a tool with the manager.

        Args:
            tool_metadata: Metadata for the tool to register
        """
        self.tools[tool_metadata.name] = tool_metadata
        logger.debug("Registered tool: %s", tool_metadata.name)

    def register_tools(self, tools: Dict[str, "ToolMetadata"]) -> None:
        """
        Register multiple tools with the manager.

        Args:
            tools: Dictionary of tool metadata objects
        """
        for _, tool in tools.items():
            self.register_tool(tool)

        logger.info("Registered %d tools", len(tools))

    def get_tool(self, tool_name: str) -> Optional["ToolMetadata"]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool metadata if found, None otherwise
        """
        return self.tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, "ToolMetadata"]:
        """
        Get all registered tools.

        Returns:
            Dictionary of all registered tools
        """
        return self.tools

    def get_enabled_tools(self) -> Dict[str, "ToolMetadata"]:
        """
        Get all enabled tools.

        Returns:
            Dictionary of enabled tools
        """
        return {name: tool for name, tool in self.tools.items() if tool.enabled}

    async def execute_tool(
        self,
        tool_name: str,
        skip_if_response_contains: Optional[List[str]] = None,
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

        Raises:
            ToolError: If the tool execution fails
            OperationTimeoutError: If the tool execution times out
        """
        if tool_name not in self.tools:
            raise ToolError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        if not tool.enabled:
            raise ToolError(f"Tool '{tool_name}' is disabled")

        # Check if we should skip the tool execution based on LLM response
        if skip_if_response_contains and hasattr(self.agent, "last_response"):
            last_response = getattr(self.agent, "last_response", "")
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
                # First, check if result is in cache
                cache_ttl = getattr(tool, "cache_ttl", 0)
                if cache_ttl > 0:
                    # Create a cache key from the tool name and arguments
                    cache_key = {
                        "tool": tool_name,
                        "args": kwargs,
                    }
                    cached_result = await default_cache_manager.get(cache_key)
                    if cached_result is not None:
                        logger.info("Using cached result for tool '%s'", tool_name)
                        return cached_result

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

    def reset(self) -> None:
        """Reset the tool manager state for a new session."""
        self.tools_used = []
