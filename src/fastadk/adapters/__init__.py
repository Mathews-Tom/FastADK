"""
Adapters for integrating FastADK agents with external platforms and services.

This module provides adapters that connect FastADK agents to various external
platforms like messaging services, APIs, and user interfaces.
"""

from .discord import DiscordAgentAdapter, create_discord_agent
from .slack import SlackAgentAdapter, create_slack_agent

__all__ = [
    "DiscordAgentAdapter",
    "SlackAgentAdapter",
    "create_discord_agent",
    "create_slack_agent",
]
