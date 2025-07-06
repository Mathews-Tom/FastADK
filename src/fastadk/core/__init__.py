"""
Core FastADK module containing base classes, decorators, and fundamental abstractions.
"""

# Agent-related components (base classes, decorators)
from .agent import Agent, BaseAgent, tool

# Batch processing utilities
from .batch import BatchResult, BatchUtils

# Cache management
from .cache import CacheManager, InMemoryCache, RedisCache, cached

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
    OrchestrationError,
    PluginError,
    ToolError,
    ValidationError,
)

# Multi-agent orchestration
from .orchestration import OrchestrationResult, OrchestrationStrategy, Orchestrator

# Plugin management system
from .plugin_manager import (
    PluginInfo,
    PluginManager,
    PluginType,
    default_plugin_manager,
)

# Summarization of conversation history
from .summarization import LLMSummarizer, SummarizationOptions, SummarizationService

# Workflow orchestration
from .workflow import (
    Workflow,
    WorkflowResult,
    WorkflowStep,
    conditional,
    merge,
    step,
    transform,
)

__all__ = [
    # Agent components
    "Agent",
    "BaseAgent",
    "tool",
    # Batch processing
    "BatchUtils",
    "BatchResult",
    # Cache management
    "CacheManager",
    "cached",
    "InMemoryCache",
    "RedisCache",
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
    # Orchestration
    "Orchestrator",
    "OrchestrationResult",
    "OrchestrationStrategy",
    # Plugin management
    "PluginManager",
    "PluginInfo",
    "PluginType",
    "default_plugin_manager",
    # Summarization
    "SummarizationService",
    "SummarizationOptions",
    "LLMSummarizer",
    # Workflow
    "Workflow",
    "WorkflowStep",
    "WorkflowResult",
    "step",
    "transform",
    "merge",
    "conditional",
    # Exceptions
    "FastADKError",
    "AgentError",
    "ConfigurationError",
    "MemoryBackendError",
    "OrchestrationError",
    "PluginError",
    "ToolError",
    "ValidationError",
]
