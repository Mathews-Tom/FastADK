"""
Plugin system for FastADK.

This module provides a plugin manager that can discover, load, and manage plugins
for the FastADK framework. It supports model providers, memory backends, tools,
and event-based plugins.
"""

import asyncio
import importlib
import inspect
import logging
import pkgutil
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

from ..memory.base import MemoryBackend
from ..providers.base import ModelProviderABC
from .plugin import EventHandler, Plugin

logger = logging.getLogger("fastadk.plugins")

T = TypeVar("T")


class PluginType(Enum):
    """Types of plugins supported by FastADK."""

    MODEL_PROVIDER = "model_provider"
    MEMORY_BACKEND = "memory_backend"
    TOOL = "tool"
    EVENT_PLUGIN = "event_plugin"
    CUSTOM = "custom"


class PluginInfo:
    """Information about a discovered plugin."""

    def __init__(
        self,
        name: str,
        plugin_type: PluginType,
        module_path: str,
        class_name: Optional[str] = None,
        version: str = "unknown",
        description: str = "",
    ) -> None:
        """
        Initialize plugin information.

        Args:
            name: The plugin name
            plugin_type: The type of plugin
            module_path: The import path to the module
            class_name: The class name within the module (if applicable)
            version: The plugin version
            description: A description of the plugin
        """
        self.name = name
        self.plugin_type = plugin_type
        self.module_path = module_path
        self.class_name = class_name
        self.version = version
        self.description = description

    def __str__(self) -> str:
        """Return a string representation of the plugin."""
        return f"{self.name} ({self.plugin_type.value}): {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin info to a dictionary."""
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "version": self.version,
            "description": self.description,
        }


class PluginManager:
    """
    Manages FastADK plugins.

    The PluginManager discovers, loads, and manages plugins for FastADK.
    It supports model providers, memory backends, custom tools, and event-based plugins.
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self._plugins: Dict[str, PluginInfo] = {}
        self._loaded_instances: Dict[str, Any] = {}
        self._loaded_classes: Dict[str, Type[Any]] = {}
        self._plugin_entry_points = ["fastadk.plugins", "fastadk_plugins"]

        # Event system
        self._event_handlers: Dict[str, List[EventHandler]] = {}
        self._active_plugins: Dict[str, Plugin] = {}

    def discover_plugins(self, rescan: bool = False) -> List[PluginInfo]:
        """
        Discover available plugins through entry points.

        Args:
            rescan: Force rediscovery of plugins

        Returns:
            List of discovered plugin information
        """
        if self._plugins and not rescan:
            return list(self._plugins.values())

        # Clear existing plugins if rescanning
        if rescan:
            self._plugins = {}

        # Discover built-in providers
        self._discover_builtin_providers()

        # Discover via entry points
        self._discover_via_entry_points()

        # Discover in installed packages
        self._discover_in_installed_packages()

        logger.info("Discovered %d plugins", len(self._plugins))
        return list(self._plugins.values())

    def _discover_builtin_providers(self) -> None:
        """Discover built-in providers from the fastadk.providers module."""
        try:
            from .. import providers

            # Find all provider modules
            provider_modules = [
                module
                for _, module, _ in pkgutil.iter_modules(
                    providers.__path__, f"{providers.__name__}."
                )
            ]

            # Process each module
            for module_name in provider_modules:
                if module_name.endswith(".base"):
                    continue  # Skip base module

                try:
                    module = importlib.import_module(module_name)

                    # Find provider classes
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, ModelProviderABC)
                            and obj is not ModelProviderABC
                        ):
                            provider_name = getattr(obj, "provider_name", name.lower())

                            plugin_info = PluginInfo(
                                name=provider_name,
                                plugin_type=PluginType.MODEL_PROVIDER,
                                module_path=module_name,
                                class_name=name,
                                description=obj.__doc__ or f"{name} provider",
                            )

                            self._plugins[
                                f"{plugin_info.plugin_type.value}.{plugin_info.name}"
                            ] = plugin_info
                            logger.debug(
                                "Discovered built-in provider: %s", provider_name
                            )

                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"Error loading built-in provider module {module_name}: {str(e)}"
                    )

        except ImportError:
            logger.debug("No built-in providers found")

    def _discover_via_entry_points(self) -> None:
        """Discover plugins using the entry_points mechanism."""
        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata  # type: ignore

        for entry_point_group in self._plugin_entry_points:
            try:
                for entry_point in metadata.entry_points(group=entry_point_group):
                    try:
                        plugin_name = entry_point.name
                        module_path = entry_point.value.split(":")[0]

                        # Try to determine plugin type from entry point name
                        plugin_type = PluginType.CUSTOM
                        if plugin_name.startswith("provider_"):
                            plugin_type = PluginType.MODEL_PROVIDER
                            plugin_name = plugin_name[9:]  # Remove "provider_" prefix
                        elif plugin_name.startswith("memory_"):
                            plugin_type = PluginType.MEMORY_BACKEND
                            plugin_name = plugin_name[7:]  # Remove "memory_" prefix
                        elif plugin_name.startswith("tool_"):
                            plugin_type = PluginType.TOOL
                            plugin_name = plugin_name[5:]  # Remove "tool_" prefix

                        # Load plugin module to get more information
                        plugin_module = importlib.import_module(module_path)
                        version = getattr(plugin_module, "__version__", "unknown")
                        description = getattr(plugin_module, "__doc__", "")

                        # Find the class in the module
                        class_name = None
                        if ":" in entry_point.value:
                            class_name = entry_point.value.split(":")[-1]

                        plugin_info = PluginInfo(
                            name=plugin_name,
                            plugin_type=plugin_type,
                            module_path=module_path,
                            class_name=class_name,
                            version=version,
                            description=description,
                        )

                        self._plugins[
                            f"{plugin_info.plugin_type.value}.{plugin_info.name}"
                        ] = plugin_info
                        logger.debug(
                            "Discovered plugin via entry point: %s", plugin_name
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error loading plugin {entry_point.name}: {str(e)}"
                        )

            except Exception as e:
                logger.warning(
                    f"Error loading entry points for group {entry_point_group}: {str(e)}"
                )

    def _discover_in_installed_packages(self) -> None:
        """Discover plugins in installed packages with 'fastadk_' prefix."""
        # Get all modules
        all_modules = set(sys.modules.keys())

        # Find modules with fastadk_ prefix
        fastadk_modules = {m for m in all_modules if m.startswith("fastadk_")}

        # Check modules for plugins
        for module_name in fastadk_modules:
            try:
                module = importlib.import_module(module_name)

                # Check if this is a plugin module
                if hasattr(module, "FASTADK_PLUGINS"):
                    plugins = getattr(module, "FASTADK_PLUGINS", [])

                    for plugin_data in plugins:
                        try:
                            plugin_name = plugin_data.get(
                                "name", module_name.split("_")[-1]
                            )
                            plugin_type_str = plugin_data.get("type", "custom")
                            plugin_type = PluginType(plugin_type_str)
                            class_name = plugin_data.get("class", None)
                            version = plugin_data.get(
                                "version", getattr(module, "__version__", "unknown")
                            )
                            description = plugin_data.get(
                                "description", getattr(module, "__doc__", "")
                            )

                            plugin_info = PluginInfo(
                                name=plugin_name,
                                plugin_type=plugin_type,
                                module_path=module_name,
                                class_name=class_name,
                                version=version,
                                description=description,
                            )

                            self._plugins[
                                f"{plugin_info.plugin_type.value}.{plugin_info.name}"
                            ] = plugin_info
                            logger.debug(
                                "Discovered plugin in package: %s", plugin_name
                            )

                        except Exception as e:
                            logger.warning(
                                f"Error processing plugin data in {module_name}: {str(e)}"
                            )

            except ImportError as e:
                logger.debug(
                    f"Could not import potential plugin module {module_name}: {str(e)}"
                )

    def load_plugin_class(self, plugin_id: str) -> Type[Any]:
        """
        Load a plugin class.

        Args:
            plugin_id: The plugin identifier

        Returns:
            The plugin class

        Raises:
            ValueError: If the plugin is not found
            ImportError: If the plugin cannot be loaded
        """
        # Check if already loaded
        if plugin_id in self._loaded_classes:
            return self._loaded_classes[plugin_id]

        # Get plugin info
        plugin_info = self._plugins.get(plugin_id)
        if not plugin_info:
            # Try with plugin type prefix
            for plugin_type in PluginType:
                full_id = f"{plugin_type.value}.{plugin_id}"
                if full_id in self._plugins:
                    plugin_info = self._plugins[full_id]
                    break

            if not plugin_info:
                available = ", ".join(self._plugins.keys())
                raise ValueError(
                    f"Plugin '{plugin_id}' not found. Available plugins: {available}"
                )

        try:
            # Import the module
            module = importlib.import_module(plugin_info.module_path)

            # Find the class
            if plugin_info.class_name:
                plugin_class = getattr(module, plugin_info.class_name)
            else:
                # Find class by convention or metadata
                if plugin_info.plugin_type == PluginType.MODEL_PROVIDER:
                    # Look for classes that inherit from ModelProviderABC
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, ModelProviderABC)
                            and obj is not ModelProviderABC
                        ):
                            plugin_class = obj
                            break
                    else:
                        raise ImportError(
                            f"No ModelProviderABC implementation found in {plugin_info.module_path}"
                        )

                elif plugin_info.plugin_type == PluginType.MEMORY_BACKEND:
                    # Look for classes that inherit from MemoryBackend
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, MemoryBackend)
                            and obj is not MemoryBackend
                        ):
                            plugin_class = obj
                            break
                    else:
                        raise ImportError(
                            f"No MemoryBackend implementation found in {plugin_info.module_path}"
                        )

                else:
                    # For other types, look for PLUGIN_CLASS attribute
                    if hasattr(module, "PLUGIN_CLASS"):
                        plugin_class = getattr(module, "PLUGIN_CLASS")
                    else:
                        # Try with capitalized name
                        capitalized = "".join(
                            word.capitalize() for word in plugin_info.name.split("_")
                        )
                        if hasattr(module, capitalized):
                            plugin_class = getattr(module, capitalized)
                        else:
                            raise ImportError(
                                f"Could not find plugin class for {plugin_id} in {plugin_info.module_path}"
                            )

            # Store the class
            self._loaded_classes[plugin_id] = plugin_class
            return plugin_class

        except (ImportError, AttributeError) as e:
            logger.error("Failed to load plugin '%s': %s", plugin_id, str(e))
            raise ImportError(f"Failed to load plugin '{plugin_id}': {str(e)}") from e

    def get_instance(
        self, plugin_id: str, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get or create an instance of a plugin.

        Args:
            plugin_id: The plugin identifier
            config: Optional configuration for the plugin

        Returns:
            An instance of the plugin

        Raises:
            ValueError: If the plugin is not found
            ImportError: If the plugin cannot be loaded
        """
        # Check if already loaded
        if plugin_id in self._loaded_instances:
            return self._loaded_instances[plugin_id]

        # Load the plugin class
        plugin_class = self.load_plugin_class(plugin_id)

        # Create an instance
        try:
            plugin_instance = plugin_class()

            # Initialize if needed
            if hasattr(plugin_instance, "initialize") and callable(
                getattr(plugin_instance, "initialize")
            ):
                # Handle async initialize
                import asyncio

                initialize_method = getattr(plugin_instance, "initialize")
                if asyncio.iscoroutinefunction(initialize_method):
                    # Run initialize in event loop if possible
                    if config:
                        try:
                            loop = asyncio.get_event_loop()
                            loop.run_until_complete(initialize_method(config))
                        except RuntimeError:
                            # No event loop - can't initialize now, will need to be done later
                            logger.warning(
                                f"Plugin '{plugin_id}' requires async initialization. "
                                "Call 'await plugin.initialize(config)' before use."
                            )
                else:
                    # Synchronous initialize
                    if config:
                        initialize_method(config)

            # Store the instance
            self._loaded_instances[plugin_id] = plugin_instance
            return plugin_instance

        except Exception as e:
            logger.error(
                "Failed to create instance of plugin '%s': %s", plugin_id, str(e)
            )
            raise ValueError(
                f"Failed to create instance of plugin '{plugin_id}': {str(e)}"
            ) from e

    def get_provider(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> ModelProviderABC:
        """
        Get a model provider instance.

        Args:
            name: The provider name
            config: Optional configuration for the provider

        Returns:
            An instance of ModelProviderABC

        Raises:
            ValueError: If the provider is not found
        """
        provider_id = f"{PluginType.MODEL_PROVIDER.value}.{name}"
        if provider_id not in self._plugins:
            provider_id = name  # Try without type prefix

        provider = self.get_instance(provider_id, config)
        return cast(ModelProviderABC, provider)

    def get_memory_backend(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> MemoryBackend:
        """
        Get a memory backend instance.

        Args:
            name: The backend name
            config: Optional configuration for the backend

        Returns:
            An instance of MemoryBackend

        Raises:
            ValueError: If the backend is not found
        """
        backend_id = f"{PluginType.MEMORY_BACKEND.value}.{name}"
        if backend_id not in self._plugins:
            backend_id = name  # Try without type prefix

        backend = self.get_instance(backend_id, config)
        return cast(MemoryBackend, backend)

    def get_providers_list(self) -> List[str]:
        """
        Get a list of available model providers.

        Returns:
            List of provider names
        """
        return [
            p.name
            for p in self._plugins.values()
            if p.plugin_type == PluginType.MODEL_PROVIDER
        ]

    def get_memory_backends_list(self) -> List[str]:
        """
        Get a list of available memory backends.

        Returns:
            List of backend names
        """
        return [
            p.name
            for p in self._plugins.values()
            if p.plugin_type == PluginType.MEMORY_BACKEND
        ]

    def get_tools_list(self) -> List[str]:
        """
        Get a list of available tools.

        Returns:
            List of tool names
        """
        return [
            p.name for p in self._plugins.values() if p.plugin_type == PluginType.TOOL
        ]

    def get_all_plugins_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all discovered plugins.

        Returns:
            List of dictionaries with plugin information
        """
        return [p.to_dict() for p in self._plugins.values()]

    # Event system methods
    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """
        Register an event handler.

        Args:
            event_name: The name of the event
            handler: The event handler function
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        logger.debug("Registered handler for event %s", event_name)

    async def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered handlers.

        Args:
            event_name: The name of the event
            event_data: The event data
        """
        # Log event (debug level)
        logger.debug("Emitting event %s with data: %s", event_name, event_data)

        # Call handlers registered directly with the plugin manager
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    result = handler(event_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(
                        "Error in event handler for %s: %s",
                        event_name,
                        e,
                        exc_info=True,
                    )

        # Call plugin handlers
        for plugin in self._active_plugins.values():
            await plugin.handle_event(event_name, event_data)

    async def register_plugin_instance(self, plugin: Plugin) -> None:
        """
        Register a plugin instance.

        Args:
            plugin: The plugin instance
        """
        if plugin.name in self._active_plugins:
            logger.warning("Plugin %s already registered, replacing", plugin.name)

        # Initialize the plugin
        await plugin.initialize(self)

        # Store the plugin
        self._active_plugins[plugin.name] = plugin
        logger.info("Registered plugin %s v%s", plugin.name, plugin.version)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a registered plugin by name.

        Args:
            name: The name of the plugin

        Returns:
            The plugin instance if found, None otherwise
        """
        return self._active_plugins.get(name)

    def get_all_plugins(self) -> List[Plugin]:
        """
        Get all registered plugins.

        Returns:
            List of plugin instances
        """
        return list(self._active_plugins.values())

    async def shutdown_plugins(self) -> None:
        """Shut down all registered plugins."""
        for plugin in self._active_plugins.values():
            try:
                await plugin.shutdown()
                logger.debug("Plugin %s shutdown complete", plugin.name)
            except Exception as e:
                logger.error("Error shutting down plugin %s: %s", plugin.name, e)

        # Clear the plugins
        self._active_plugins.clear()


# Singleton instance
default_plugin_manager = PluginManager()
