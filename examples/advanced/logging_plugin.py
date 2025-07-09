"""
Example Logging Plugin for FastADK.

This module demonstrates how to create a plugin using the new event-based
plugin system in FastADK.

Usage:
    1. Import the LoggingPlugin class
    2. Create an instance of the plugin
    3. Register it with your agent

Example:
    from fastadk import Agent, BaseAgent
    from fastadk.core.plugin import Plugin
    from examples.advanced.logging_plugin import LoggingPlugin

    @Agent(model="gpt-4", description="Example agent")
    class MyAgent(BaseAgent):
        def __init__(self):
            super().__init__()

            # Register the logging plugin
            plugin = LoggingPlugin()
            self.register_plugin(plugin)
"""

import logging
import time
from typing import Any, Dict

from fastadk.core.plugin import Plugin


class LoggingPlugin(Plugin):
    """A plugin that logs agent activities and events."""

    def __init__(self) -> None:
        """Initialize the plugin."""
        super().__init__(
            name="logging_plugin",
            version="1.0.0",
            description="Logs agent activities and events",
        )
        self.logger = logging.getLogger("fastadk.plugins.logging")
        self.events_logged = 0
        self.start_time = time.time()
        self.handler_registered = False

    async def initialize(self, plugin_manager, **kwargs) -> None:
        """Initialize the plugin.

        Args:
            plugin_manager: The plugin manager instance
            **kwargs: Additional keyword arguments
        """
        await super().initialize(plugin_manager, **kwargs)
        self.logger.info("Initializing %s v%s", self.name, self.version)

        if not self.handler_registered:
            self.register_event_handlers()
            self.handler_registered = True

    def register_event_handlers(self) -> None:
        """Register event handlers for this plugin."""
        self.register_event_handler("agent:created", self.on_agent_created)
        self.register_event_handler("agent:initialized", self.on_agent_initialized)
        self.register_event_handler("agent:run_started", self.on_agent_run_started)
        self.register_event_handler("agent:run_completed", self.on_agent_run_completed)
        self.register_event_handler("agent:run_error", self.on_agent_run_error)
        self.register_event_handler("tool:called", self.on_tool_called)
        self.register_event_handler("tool:completed", self.on_tool_completed)
        self.register_event_handler("llm:request", self.on_llm_request)
        self.register_event_handler("llm:response", self.on_llm_response)

    async def on_agent_created(self, event_data: Dict[str, Any]) -> None:
        """Handle agent creation event."""
        agent_class = event_data.get("agent_class", "Unknown")
        agent_id = id(event_data.get("agent", None))
        self.logger.info("Agent created: %s (id: %s)", agent_class, agent_id)
        self.events_logged += 1

    async def on_agent_initialized(self, event_data: Dict[str, Any]) -> None:
        """Handle agent initialization event."""
        agent = event_data.get("agent", None)
        agent_class = event_data.get("agent_class", "Unknown")
        agent_id = id(agent) if agent else "Unknown"
        self.logger.info("Agent initialized: %s (id: %s)", agent_class, agent_id)
        self.events_logged += 1

    async def on_agent_run_started(self, event_data: Dict[str, Any]) -> None:
        """Handle agent run started event."""
        agent_class = event_data.get("agent_class", "Unknown")
        input_text = event_data.get("input", "")
        truncated_input = (
            (input_text[:50] + "...") if len(input_text) > 50 else input_text
        )
        self.logger.info("Agent %s run started: '%s'", agent_class, truncated_input)
        self.events_logged += 1

    async def on_agent_run_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle agent run completed event."""
        agent_class = event_data.get("agent_class", "Unknown")
        duration = event_data.get("duration", 0)
        self.logger.info("Agent %s run completed in %.2fs", agent_class, duration)
        self.events_logged += 1

    async def on_agent_run_error(self, event_data: Dict[str, Any]) -> None:
        """Handle agent run error event."""
        agent_class = event_data.get("agent_class", "Unknown")
        error = event_data.get("error", "Unknown error")
        self.logger.info("Agent %s run failed: %s", agent_class, error)
        self.events_logged += 1

    async def on_tool_called(self, event_data: Dict[str, Any]) -> None:
        """Handle tool called event."""
        tool_name = event_data.get("tool_name", "Unknown")
        args = event_data.get("args", {})
        self.logger.info("Tool called: %s with args: %s", tool_name, args)
        self.events_logged += 1

    async def on_tool_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle tool completed event."""
        tool_name = event_data.get("tool_name", "Unknown")
        duration = event_data.get("duration", 0)
        success = event_data.get("success", False)
        status = "succeeded" if success else "failed"
        self.logger.info("Tool %s %s in %.2fs", tool_name, status, duration)
        self.events_logged += 1

    async def on_llm_request(self, event_data: Dict[str, Any]) -> None:
        """Handle LLM request event."""
        model = event_data.get("model", "Unknown")
        temperature = event_data.get("temperature", 0)
        max_tokens = event_data.get("max_tokens", 0)
        self.logger.info(
            "LLM request sent to model: %s (temp: %.2f, max_tokens: %d)",
            model,
            temperature,
            max_tokens,
        )
        self.events_logged += 1

    async def on_llm_response(self, event_data: Dict[str, Any]) -> None:
        """Handle LLM response event."""
        model = event_data.get("model", "Unknown")
        tokens = event_data.get("tokens", 0)
        duration = event_data.get("duration", 0)
        self.logger.info(
            "LLM response received from model: %s (%d tokens in %.2fs)",
            model,
            tokens,
            duration,
        )
        self.events_logged += 1

    async def shutdown(self) -> None:
        """Clean up resources when plugin is deactivated."""
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            "Shutting down %s. Logged %d events in %.2fs",
            self.name,
            self.events_logged,
            elapsed_time,
        )
