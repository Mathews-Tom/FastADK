"""
Core FastADK module containing base classes, decorators, and fundamental abstractions.
"""

# Agent-related components (base classes, decorators)
from .agent import Agent, BaseAgent, tool

# Configuration management
from .config import FastADKSettings, get_settings, reload_settings

# Context management for conversations
from .context import ContextEntry, ContextManager, ContextWindow, ConversationContext

# Context policies for controlling conversation history
from .context_policy import (
    ContextPolicy,
    HybridVectorRetrievalPolicy,
    MostRecentPolicy,
    SummarizeOlderPolicy,
)

# Exception handling and error types
from .exceptions import (
    AgentError,
    ConfigurationError,
    FastADKError,
    MemoryBackendError,
    PluginError,
    ToolError,
    ValidationError,
)

# Summarization of conversation history
from .summarization import LLMSummarizer, SummarizationOptions, SummarizationService

__all__ = [
    # Agent components
    "Agent",
    "BaseAgent",
    "tool",
    # Configuration
    "FastADKSettings",
    "get_settings",
    "reload_settings",
    # Context management
    "ConversationContext",
    "ContextEntry",
    "ContextManager",
    "ContextWindow",
    # Context policies
    "ContextPolicy",
    "MostRecentPolicy",
    "SummarizeOlderPolicy",
    "HybridVectorRetrievalPolicy",
    # Summarization
    "SummarizationService",
    "SummarizationOptions",
    "LLMSummarizer",
    # Exceptions
    "FastADKError",
    "AgentError",
    "ConfigurationError",
    "MemoryBackendError",
    "PluginError",
    "ToolError",
    "ValidationError",
]
