"""
Plugin base class for FastADK.

This module provides the base class for plugins in FastADK along with
related utilities for plugin development.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Type, TypeVar, Union

logger = logging.getLogger("fastadk.plugins")

# Type definitions
EventHandler = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
PluginManagerType = TypeVar("PluginManagerType")


class Plugin:
    """Base class for all FastADK plugins."""

    def __init__(
        self, name: str, version: str = "0.1.0", description: str = ""
    ) -> None:
        """Initialize the plugin.

        Args:
            name: The name of the plugin
            version: The version of the plugin
            description: A description of the plugin
        """
        self.name = name
        self.version = version
        self.description = description
        self.initialized = False
        self._event_handlers: Dict[str, List[EventHandler]] = {}

    async def initialize(
        self, plugin_manager: Any = None, **kwargs
    ) -> None:  # pylint: disable=unused-argument
        """Initialize the plugin.

        This method is called when the plugin is first registered with a plugin manager.
        Override this method to perform any setup that requires a reference to the
        plugin manager.

        Args:
            plugin_manager: The plugin manager instance
            **kwargs: Additional keyword arguments
        """
        # Store reference to plugin manager if needed in subclasses
        # Parameters are unused in base class but may be used by subclasses
        self.initialized = True

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated.

        Override this method to perform any cleanup when the plugin is deactivated.
        """
        self.initialized = False

    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """Register an event handler for this plugin.

        Args:
            event_name: The name of the event
            handler: The event handler function
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        logger.debug("Plugin %s registered handler for event %s", self.name, event_name)

    async def handle_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Handle an event.

        This method is called by the plugin manager when an event occurs.
        It calls all registered handlers for the specified event.

        Args:
            event_name: The name of the event
            event_data: The event data
        """
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    result = handler(event_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(
                        "Error in plugin %s while handling event %s: %s",
                        self.name,
                        event_name,
                        exc,
                        exc_info=True,
                    )

    def __str__(self) -> str:
        """Return a string representation of the plugin."""
        return f"{self.name} v{self.version}: {self.description}"


class AgentPlugin(Plugin):
    """Base class for plugins that enhance agent functionality."""

    async def on_agent_initialized(self, agent: Any) -> None:
        """Called when an agent is initialized.

        Override this method to enhance agent functionality when it's initialized.

        Args:
            agent: The agent instance
        """
        # This method should be overridden by subclasses
        # No implementation needed in base class


class PluginRegistry:
    """Registry for plugin-enabled classes.

    This class is used to mark a class as plugin-enabled and provide
    metadata about supported plugins.
    """

    def __init__(self, cls: Type) -> None:
        """Initialize the plugin registry.

        Args:
            cls: The class to register
        """
        self.cls = cls
        self.supported_plugins: List[Type[Plugin]] = []

    def supports_plugin(self, plugin_cls: Type[Plugin]) -> None:
        """Mark the class as supporting a specific plugin type.

        Args:
            plugin_cls: The plugin class
        """
        self.supported_plugins.append(plugin_cls)


# Registry of plugin-enabled classes
_plugin_registry: Dict[Type, PluginRegistry] = {}


def plugin_enabled(cls: Type) -> Type:
    """Decorator to mark a class as plugin-enabled.

    Args:
        cls: The class to mark as plugin-enabled

    Returns:
        The decorated class
    """
    if cls not in _plugin_registry:
        _plugin_registry[cls] = PluginRegistry(cls)
    return cls


def supports_plugin(plugin_cls: Type[Plugin]) -> Callable[[Type], Type]:
    """Decorator to mark a class as supporting a specific plugin type.

    Args:
        plugin_cls: The plugin class

    Returns:
        Decorator function
    """

    def decorator(cls: Type) -> Type:
        """Decorator implementation.

        Args:
            cls: The class to decorate

        Returns:
            The decorated class
        """
        if cls not in _plugin_registry:
            _plugin_registry[cls] = PluginRegistry(cls)
        _plugin_registry[cls].supports_plugin(plugin_cls)
        return cls

    return decorator


def get_supported_plugins(cls: Type) -> List[Type[Plugin]]:
    """Get the plugins supported by a class.

    Args:
        cls: The class to check

    Returns:
        List of supported plugin classes
    """
    if cls in _plugin_registry:
        return _plugin_registry[cls].supported_plugins
    return []
