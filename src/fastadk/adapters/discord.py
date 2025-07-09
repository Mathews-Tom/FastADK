"""
Discord adapter for FastADK agents.

This module provides a Discord adapter that allows FastADK agents to integrate
with Discord servers, responding to messages in channels and direct messages.
"""

import asyncio
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..core.agent import BaseAgent
from ..core.exceptions import ConfigurationError

logger = logging.getLogger("fastadk.adapters.discord")


class DiscordAgentAdapter:
    """
    Adapts FastADK agents to work with Discord.

    This adapter connects FastADK agents to Discord, allowing them to respond
    to messages in channels and direct messages.
    """

    def __init__(
        self,
        agent: BaseAgent,
        token: Optional[str] = None,
        guild_ids: Optional[List[int]] = None,
        channel_ids: Optional[List[int]] = None,
        mention_only: bool = True,
        include_thread_history: bool = True,
        max_history_messages: int = 10,
        command_prefix: str = "!",
    ) -> None:
        """
        Initialize the Discord adapter.

        Args:
            agent: The FastADK agent to adapt
            token: Discord bot token (or set DISCORD_BOT_TOKEN env var)
            guild_ids: List of guild (server) IDs the bot should operate in (empty = all)
            channel_ids: List of channel IDs the bot should respond in (empty = all)
            mention_only: Only respond when the bot is mentioned or command used
            include_thread_history: Include previous messages as context
            max_history_messages: Maximum number of history messages to include
            command_prefix: Prefix for bot commands (default: !)

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self.agent = agent
        self.token = token or os.environ.get("DISCORD_BOT_TOKEN")
        self.guild_ids = guild_ids or []
        self.channel_ids = channel_ids or []
        self.mention_only = mention_only
        self.include_thread_history = include_thread_history
        self.max_history_messages = max_history_messages
        self.command_prefix = command_prefix

        self._client = None
        self._bot_id = None
        self._message_cache: Dict[int, List[Dict[str, Any]]] = {}

        # Validate configuration
        if not self.token:
            raise ConfigurationError(
                "Discord bot token is required. Either pass it as a parameter or "
                "set the DISCORD_BOT_TOKEN environment variable."
            )

    async def initialize(self) -> None:
        """
        Initialize the Discord adapter.

        This method sets up the Discord client, including registering event handlers.

        Raises:
            ConfigurationError: If discord.py is not installed or initialization fails
        """
        try:
            # We import here to avoid hard dependency
            import discord
            from discord.ext import commands

            # Set up intents
            intents = discord.Intents.default()
            intents.messages = True
            intents.message_content = True

            # Create client
            self._client = commands.Bot(
                command_prefix=self.command_prefix, intents=intents
            )

            # Register event handlers
            @self._client.event
            async def on_ready():
                self._bot_id = self._client.user.id
                logger.info(
                    "Discord adapter initialized for agent: %s (connected as %s)",
                    self.agent.__class__.__name__,
                    self._client.user.name,
                )

            @self._client.event
            async def on_message(message):
                await self._handle_message(message)
                await self._client.process_commands(message)

            # Register help command
            @self._client.command(name="help")
            async def help_command(ctx):
                """Get help about using the agent."""
                help_text = (
                    f"**{self.agent.__class__.__name__} Bot**\n\n"
                    f"{self.agent.__doc__ or 'No description available.'}\n\n"
                    f"You can interact with me by:\n"
                    f"- Mentioning me: @{self._client.user.name} <your message>\n"
                    f"- Using the command: {self.command_prefix}ask <your message>\n"
                )
                await ctx.send(help_text)

            # Register ask command
            @self._client.command(name="ask")
            async def ask_command(ctx, *, question):
                """Ask the agent a question."""
                # Get message history if enabled
                message_context = ""
                if self.include_thread_history:
                    history = await self._get_conversation_history(
                        ctx.channel, ctx.message
                    )
                    message_context = self._format_conversation_history(history)

                # Combine context and current message
                full_input = (
                    f"{message_context}\n\n{question}" if message_context else question
                )

                # Show typing indicator
                async with ctx.typing():
                    try:
                        response = await self.agent.run(full_input)

                        # Handle Discord's 2000 character limit
                        chunks = self._chunk_message(response)
                        for chunk in chunks:
                            await ctx.send(chunk)

                    except Exception as e:
                        logger.error("Error processing command with agent: %s", str(e))
                        await ctx.send(f"Sorry, I encountered an error: {str(e)}")

            logger.info(
                "Discord adapter initialized for agent: %s",
                self.agent.__class__.__name__,
            )

        except ImportError:
            raise ConfigurationError(
                "discord.py package is required. Install with: uv add discord.py"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Discord adapter: {str(e)}")

    async def start(self) -> None:
        """
        Start the Discord bot.

        This method starts the Discord client and connects to Discord.

        Raises:
            ConfigurationError: If the client failed to initialize or start
        """
        if not self._client:
            await self.initialize()

        try:
            await self._client.start(self.token)

        except Exception as e:
            raise ConfigurationError(f"Failed to start Discord adapter: {str(e)}")

    async def _handle_message(self, message) -> None:
        """
        Handle a Discord message.

        Args:
            message: The Discord message object
        """
        # Skip messages from the bot itself
        if message.author.id == self._bot_id:
            return

        # Skip messages in guilds we're not monitoring (if specified)
        if self.guild_ids and (
            not message.guild or message.guild.id not in self.guild_ids
        ):
            return

        # Skip messages in channels we're not monitoring (if specified)
        if self.channel_ids and message.channel.id not in self.channel_ids:
            return

        # Cache the message for history
        self._cache_message(message)

        # If mention_only is True, only respond when mentioned or command used
        if self.mention_only:
            # Check if bot was mentioned
            bot_mentioned = self._bot_id in [user.id for user in message.mentions]
            # Check if message starts with command prefix (but is not a command)
            is_command_like = message.content.startswith(
                self.command_prefix
            ) and not message.content.startswith(f"{self.command_prefix}ask")

            if not bot_mentioned and not is_command_like:
                return

        # Get conversation context
        message_context = ""
        if self.include_thread_history:
            history = await self._get_conversation_history(message.channel, message)
            message_context = self._format_conversation_history(history)

        # Clean text (remove mentions, etc.)
        clean_text = self._clean_message_text(message.content)

        # Combine context and current message
        full_input = (
            f"{message_context}\n\n{clean_text}" if message_context else clean_text
        )

        # Process with the agent
        async with message.channel.typing():
            try:
                response = await self.agent.run(full_input)

                # Handle Discord's 2000 character limit
                chunks = self._chunk_message(response)
                for chunk in chunks:
                    await message.channel.send(chunk, reference=message)

                logger.info("Sent response to Discord channel %s", message.channel.id)

            except Exception as e:
                logger.error("Error processing message with agent: %s", str(e))
                # Send error message
                await message.channel.send(
                    f"Sorry, I encountered an error: {str(e)}", reference=message
                )

    def _cache_message(self, message) -> None:
        """
        Cache a message for conversation history.

        Args:
            message: The Discord message to cache
        """
        channel_id = message.channel.id
        if channel_id not in self._message_cache:
            self._message_cache[channel_id] = []

        # Add message to cache
        self._message_cache[channel_id].append(
            {
                "id": message.id,
                "author_id": message.author.id,
                "author_name": message.author.name,
                "content": message.content,
                "timestamp": message.created_at.timestamp(),
                "is_bot": message.author.bot,
            }
        )

        # Limit cache size
        if len(self._message_cache[channel_id]) > self.max_history_messages * 2:
            self._message_cache[channel_id] = self._message_cache[channel_id][
                -self.max_history_messages :
            ]

    async def _get_conversation_history(
        self, channel, current_message
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history from a channel.

        Args:
            channel: The Discord channel
            current_message: The current message

        Returns:
            List of message data dictionaries
        """
        try:
            # First check cache
            if channel.id in self._message_cache:
                # Get messages before the current one
                history = [
                    msg
                    for msg in self._message_cache[channel.id]
                    if msg["timestamp"] < current_message.created_at.timestamp()
                ]
                # Return most recent messages up to limit
                return history[-self.max_history_messages :]

            # If not in cache, fetch from Discord
            messages = []
            async for msg in channel.history(
                limit=self.max_history_messages, before=current_message
            ):
                messages.append(
                    {
                        "id": msg.id,
                        "author_id": msg.author.id,
                        "author_name": msg.author.name,
                        "content": msg.content,
                        "timestamp": msg.created_at.timestamp(),
                        "is_bot": msg.author.bot,
                    }
                )

            # Return in chronological order
            return list(reversed(messages))

        except Exception as e:
            logger.error("Error fetching message history: %s", str(e))
            return []

    def _clean_message_text(self, text: str) -> str:
        """
        Clean a message text by removing mentions and normalizing whitespace.

        Args:
            text: The message text to clean

        Returns:
            Cleaned message text
        """
        # Remove bot mentions
        if self._bot_id:
            text = re.sub(f"<@!?{self._bot_id}>", "", text)

        # Remove command prefix if present
        if text.startswith(self.command_prefix):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                text = parts[1]

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation history as context.

        Args:
            messages: List of message data dictionaries

        Returns:
            Formatted conversation history
        """
        history_lines = []

        for msg in messages:
            # Skip empty messages
            if not msg.get("content"):
                continue

            is_bot = msg.get("is_bot", False)
            text = self._clean_message_text(msg.get("content", ""))

            if is_bot and self._bot_id and msg.get("author_id") == self._bot_id:
                history_lines.append(f"Assistant: {text}")
            else:
                author = msg.get("author_name", "User")
                history_lines.append(f"{author}: {text}")

        return "\n".join(history_lines)

    def _chunk_message(self, message: str, chunk_size: int = 1990) -> List[str]:
        """
        Split a message into chunks that fit within Discord's message size limit.

        Args:
            message: The message to split
            chunk_size: Maximum size of each chunk

        Returns:
            List of message chunks
        """
        if len(message) <= chunk_size:
            return [message]

        chunks = []

        # Try to split on paragraph breaks first
        paragraphs = message.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # If adding this paragraph would exceed chunk size
                if current_chunk:
                    chunks.append(current_chunk)

                # If paragraph itself is too long, split it
                if len(paragraph) > chunk_size:
                    # Split long paragraph into sentences
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)

                            # If sentence itself is too long, split it by chunk size
                            if len(sentence) > chunk_size:
                                sentence_chunks = [
                                    sentence[i : i + chunk_size]
                                    for i in range(0, len(sentence), chunk_size)
                                ]
                                chunks.extend(sentence_chunks[:-1])
                                current_chunk = sentence_chunks[-1]
                            else:
                                current_chunk = sentence
                else:
                    current_chunk = paragraph

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def send_message(
        self, channel_id: int, text: str, reference_message_id: Optional[int] = None
    ) -> Any:
        """
        Send a message to a Discord channel.

        Args:
            channel_id: The channel ID to send to
            text: The message text
            reference_message_id: Optional message ID to reply to

        Returns:
            Discord message object

        Raises:
            ConfigurationError: If the client failed to initialize or the API call fails
        """
        if not self._client:
            await self.initialize()

        try:
            # Get the channel
            channel = self._client.get_channel(channel_id)
            if not channel:
                channel = await self._client.fetch_channel(channel_id)

            # Handle reference message if provided
            reference = None
            if reference_message_id:
                try:
                    reference_message = await channel.fetch_message(
                        reference_message_id
                    )
                    reference = reference_message
                except Exception as e:
                    logger.warning("Could not fetch reference message: %s", str(e))

            # Handle Discord's 2000 character limit
            chunks = self._chunk_message(text)
            sent_messages = []

            for i, chunk in enumerate(chunks):
                # Only reference the original message for the first chunk
                if i == 0 and reference:
                    sent_message = await channel.send(chunk, reference=reference)
                else:
                    sent_message = await channel.send(chunk)

                sent_messages.append(sent_message)

            return sent_messages[0] if sent_messages else None

        except Exception as e:
            logger.error("Error sending message to Discord: %s", str(e))
            raise ConfigurationError(f"Failed to send message to Discord: {str(e)}")


async def create_discord_agent(
    agent: BaseAgent,
    token: Optional[str] = None,
    guild_ids: Optional[List[int]] = None,
    mention_only: bool = True,
) -> DiscordAgentAdapter:
    """
    Create and initialize a Discord agent adapter.

    This is a convenience function to create, initialize, and return
    a DiscordAgentAdapter instance ready for use.

    Args:
        agent: The FastADK agent to adapt
        token: Discord bot token (or set DISCORD_BOT_TOKEN env var)
        guild_ids: List of guild (server) IDs the bot should operate in (empty = all)
        mention_only: Only respond when the bot is mentioned or command used

    Returns:
        Initialized DiscordAgentAdapter

    Raises:
        ConfigurationError: If initialization fails
    """
    adapter = DiscordAgentAdapter(
        agent=agent,
        token=token,
        guild_ids=guild_ids,
        mention_only=mention_only,
    )

    await adapter.initialize()
    return adapter
