"""
Slack adapter for FastADK agents.

This module provides a Slack adapter that allows FastADK agents to integrate
with Slack workspaces, responding to messages in channels and direct messages.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..core.agent import BaseAgent
from ..core.exceptions import ConfigurationError

logger = logging.getLogger("fastadk.adapters.slack")


class SlackAgentAdapter:
    """
    Adapts FastADK agents to work with Slack.

    This adapter connects FastADK agents to Slack, allowing them to respond
    to messages in channels and direct messages.
    """

    def __init__(
        self,
        agent: BaseAgent,
        token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        channel_ids: Optional[List[str]] = None,
        mention_only: bool = True,
        include_thread_history: bool = True,
        max_history_messages: int = 10,
    ) -> None:
        """
        Initialize the Slack adapter.

        Args:
            agent: The FastADK agent to adapt
            token: Slack bot token (or set SLACK_BOT_TOKEN env var)
            signing_secret: Slack signing secret (or set SLACK_SIGNING_SECRET env var)
            channel_ids: List of channel IDs the bot should respond in (empty = all)
            mention_only: Only respond when the bot is mentioned
            include_thread_history: Include previous messages in a thread as context
            max_history_messages: Maximum number of history messages to include

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self.agent = agent
        self.token = token or os.environ.get("SLACK_BOT_TOKEN")
        self.signing_secret = signing_secret or os.environ.get("SLACK_SIGNING_SECRET")
        self.channel_ids = channel_ids or []
        self.mention_only = mention_only
        self.include_thread_history = include_thread_history
        self.max_history_messages = max_history_messages

        self._app = None
        self._bot_id = None
        self._bot_user_id = None
        self._mention_pattern = None

        # Validate configuration
        if not self.token:
            raise ConfigurationError(
                "Slack bot token is required. Either pass it as a parameter or "
                "set the SLACK_BOT_TOKEN environment variable."
            )

        if not self.signing_secret:
            raise ConfigurationError(
                "Slack signing secret is required. Either pass it as a parameter or "
                "set the SLACK_SIGNING_SECRET environment variable."
            )

    async def initialize(self) -> None:
        """
        Initialize the Slack adapter.

        This method sets up the Slack client and app, including fetching
        bot identity information.

        Raises:
            ConfigurationError: If Slack API calls fail
        """
        try:
            # We import here to avoid hard dependency on slack_bolt
            from slack_bolt.async_app import AsyncApp

            # Initialize the Slack app
            self._app = AsyncApp(token=self.token, signing_secret=self.signing_secret)

            # Register message handlers
            self._app.event("message")(self._handle_message)
            self._app.event("app_mention")(self._handle_app_mention)

            # Get bot info
            client = self._app.client
            bot_info = await client.auth_test()
            self._bot_id = bot_info["bot_id"]
            self._bot_user_id = bot_info["user_id"]

            # Create mention pattern
            self._mention_pattern = re.compile(f"<@{self._bot_user_id}>")

            logger.info(
                "Slack adapter initialized for agent: %s", self.agent.__class__.__name__
            )

        except ImportError as exc:
            raise ConfigurationError(
                "slack_bolt package is required. Install with: uv add slack-bolt"
            ) from exc
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Slack adapter: {str(e)}"
            ) from e

    async def start(self) -> None:
        """
        Start listening for Slack events.

        This method starts the Slack app in socket mode, which allows it to
        receive events from Slack in real-time.

        Raises:
            ConfigurationError: If the app failed to initialize or start
        """
        if not self._app:
            await self.initialize()

        try:
            # Import here to avoid hard dependency
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )

            # Start the app in socket mode
            handler = AsyncSocketModeHandler(self._app, self.token)
            await handler.start_async()

            logger.info("Slack adapter started and listening for events")

        except ImportError as exc:
            raise ConfigurationError(
                "slack_bolt package is required. Install with: uv add slack-bolt"
            ) from exc
        except (ValueError, RuntimeError, ConnectionError) as e:
            raise ConfigurationError(f"Failed to start Slack adapter: {str(e)}") from e

    async def _handle_message(self, event: Dict[str, Any], client: Any) -> None:
        """
        Handle a Slack message event.

        Args:
            event: The Slack event data
            client: The Slack client
        """
        # Skip messages from the bot itself
        if event.get("user") == self._bot_user_id:
            return

        # Skip messages in channels we're not monitoring (if specified)
        channel_id = event.get("channel")
        if self.channel_ids and channel_id not in self.channel_ids:
            return

        # If mention_only is True, only respond when mentioned
        text = event.get("text", "")
        if self.mention_only and not self._mention_pattern.search(text):
            return

        # Get conversation context
        thread_ts = event.get("thread_ts", event.get("ts"))
        message_context = ""

        if self.include_thread_history and thread_ts:
            try:
                # Get thread history
                history_response = await client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=self.max_history_messages,
                )

                if history_response and history_response["ok"]:
                    # Format history as context
                    message_context = self._format_conversation_history(
                        history_response.get("messages", []),
                        event.get("ts"),
                    )

            except (ValueError, KeyError, ConnectionError) as e:
                logger.error("Error fetching thread history: %s", str(e))

        # Clean text (remove mentions, etc.)
        clean_text = self._clean_message_text(text)

        # Combine context and current message
        full_input = (
            f"{message_context}\n\n{clean_text}" if message_context else clean_text
        )

        # Process with the agent
        try:
            response = await self.agent.run(full_input)

            # Send the response
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=response,
            )

            logger.info("Sent response to Slack channel %s", channel_id)

        except (ValueError, RuntimeError, AttributeError, KeyError) as e:
            logger.error("Error processing message with agent: %s", str(e))
            # Send error message
            try:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"Sorry, I encountered an error: {str(e)}",
                )
            except (ConnectionError, KeyError) as send_error:
                logger.error(
                    "Failed to send error message to Slack: %s", str(send_error)
                )

    async def _handle_app_mention(self, event: Dict[str, Any], client: Any) -> None:
        """
        Handle when the app is directly mentioned.

        Args:
            event: The Slack event data
            client: The Slack client
        """
        # Reuse the message handler for mentions
        await self._handle_message(event, client)

    def _clean_message_text(self, text: str) -> str:
        """
        Clean a message text by removing mentions and normalizing whitespace.

        Args:
            text: The message text to clean

        Returns:
            Cleaned message text
        """
        # Remove mentions
        cleaned = re.sub(r"<@[A-Z0-9]+>", "", text)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _format_conversation_history(
        self, messages: List[Dict[str, Any]], current_message_ts: str
    ) -> str:
        """
        Format conversation history as context.

        Args:
            messages: List of messages from the thread
            current_message_ts: Timestamp of the current message to exclude

        Returns:
            Formatted conversation history
        """
        history_lines = []

        for msg in messages:
            # Skip the current message (we'll add it separately)
            if msg.get("ts") == current_message_ts:
                continue

            user = msg.get("user", "unknown")
            is_bot = user == self._bot_user_id
            text = self._clean_message_text(msg.get("text", ""))

            if is_bot:
                history_lines.append(f"Assistant: {text}")
            else:
                history_lines.append(f"User: {text}")

        return "\n".join(history_lines)

    async def send_message(
        self, channel_id: str, text: str, thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a Slack channel.

        Args:
            channel_id: The channel ID to send to
            text: The message text
            thread_ts: Optional thread timestamp to reply in a thread

        Returns:
            Slack API response

        Raises:
            ConfigurationError: If the app failed to initialize or the API call fails
        """
        if not self._app:
            await self.initialize()

        try:
            response = await self._app.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
            return response

        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.error("Error sending message to Slack: %s", str(e))
            raise ConfigurationError(
                f"Failed to send message to Slack: {str(e)}"
            ) from e


async def create_slack_agent(
    agent: BaseAgent,
    token: Optional[str] = None,
    channel_ids: Optional[List[str]] = None,
    mention_only: bool = True,
) -> SlackAgentAdapter:
    """
    Create and initialize a Slack agent adapter.

    This is a convenience function to create, initialize, and return
    a SlackAgentAdapter instance ready for use.

    Args:
        agent: The FastADK agent to adapt
        token: Slack bot token (or set SLACK_BOT_TOKEN env var)
        channel_ids: List of channel IDs the bot should respond in (empty = all)
        mention_only: Only respond when the bot is mentioned

    Returns:
        Initialized SlackAgentAdapter

    Raises:
        ConfigurationError: If initialization fails
    """
    adapter = SlackAgentAdapter(
        agent=agent,
        token=token,
        channel_ids=channel_ids,
        mention_only=mention_only,
    )

    await adapter.initialize()
    return adapter
