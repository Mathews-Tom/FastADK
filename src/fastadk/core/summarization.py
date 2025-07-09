"""
Summarization services for FastADK.

This module provides services for summarizing conversation histories
to keep context windows manageable.
"""

import abc
import time
from typing import Any, Dict, List, Optional, Protocol

from loguru import logger
from pydantic import BaseModel

from fastadk.core.context import ContextEntry
from fastadk.memory.base import MemoryBackend


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers that can be used for summarization."""

    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional arguments to pass to the provider

        Returns:
            The generated response as a string
        """
        ...


class SummarizationService(abc.ABC):
    """
    Abstract base class for summarization services.

    A summarization service is responsible for creating summaries of
    conversation histories to keep context windows manageable.
    """

    @abc.abstractmethod
    async def summarize(self, entries: List[ContextEntry]) -> ContextEntry:
        """
        Summarize a list of context entries.

        Args:
            entries: The entries to summarize

        Returns:
            A new context entry containing the summary
        """
        pass


class SummarizationOptions(BaseModel):
    """Options for controlling summarization behavior."""

    max_length: Optional[int] = None
    preserve_key_points: bool = True
    include_entities: bool = True
    style: str = "concise"  # "concise", "detailed", "bullets"
    model: Optional[str] = None  # Specific model to use for summarization


class LLMSummarizer(SummarizationService):
    """
    Summarization service that uses an LLM to generate summaries.

    This service sends conversation histories to an LLM with a prompt
    that asks for a summary, then returns the summary as a new message.
    """

    def __init__(
        self,
        llm_provider: LLMProviderProtocol,
        memory_backend: Optional[MemoryBackend] = None,
        cache_ttl: int = 3600,  # 1 hour
        default_options: Optional[SummarizationOptions] = None,
    ):
        """
        Initialize the LLM summarizer.

        Args:
            llm_provider: The LLM provider to use
            memory_backend: Optional memory backend for caching summaries
            cache_ttl: TTL for cached summaries in seconds
            default_options: Default summarization options
        """
        self.llm = llm_provider
        self.memory = memory_backend
        self.cache_ttl = cache_ttl
        self.default_options = default_options or SummarizationOptions()

    async def summarize(
        self,
        entries: List[ContextEntry],
        options: Optional[SummarizationOptions] = None,
    ) -> ContextEntry:
        """
        Summarize a list of context entries using an LLM.

        Args:
            entries: The entries to summarize
            options: Optional summarization options

        Returns:
            A new context entry containing the summary
        """
        if not entries:
            return ContextEntry(
                role="system",
                content="No previous conversation to summarize.",
            )

        # Use provided options or fall back to defaults
        opts = options or self.default_options

        # Check cache if memory backend is available
        if self.memory:
            cache_key = self._generate_cache_key(entries, opts)
            cached_entry = await self.memory.get(f"summary:{cache_key}")
            if cached_entry and cached_entry.data:
                logger.debug("Using cached summary for key %s", cache_key)
                return ContextEntry(**cached_entry.data)

        # Convert entries to a format suitable for the LLM
        formatted_entries = [
            {"role": entry.role, "content": entry.content} for entry in entries
        ]

        # Add a system message asking for a summary
        system_message = self._create_system_message(opts)
        messages = [system_message] + formatted_entries

        try:
            # Generate summary using the LLM
            summary_text = await self.llm.generate(messages=messages)

            # Create a new context entry with the summary
            summary_entry = ContextEntry(
                role="system",
                content=f"Previous conversation summary: {summary_text}",
                metadata={
                    "is_summary": True,
                    "summarized_count": len(entries),
                    "summarization_timestamp": time.time(),
                },
            )

            # Cache the summary if memory backend is available
            if self.memory:
                await self.memory.set(
                    f"summary:{cache_key}",
                    summary_entry.model_dump(),
                    ttl_seconds=self.cache_ttl,
                )

            return summary_entry
        except Exception as e:
            logger.error("Error generating summary: %s", str(e))
            # Fallback to a basic summary if LLM generation fails
            return self._create_fallback_summary(entries)

    def _create_system_message(self, options: SummarizationOptions) -> Dict[str, str]:
        """
        Create a system message that prompts the LLM to generate a summary.

        Args:
            options: Summarization options

        Returns:
            A system message dictionary
        """
        style_instructions = {
            "concise": "Keep the summary brief and to the point.",
            "detailed": "Include important details from the conversation.",
            "bullets": "Format the summary as bullet points of key information.",
        }.get(options.style, "Keep the summary brief and to the point.")

        length_instruction = ""
        if options.max_length:
            length_instruction = f" Keep the summary under {options.max_length} words."

        entity_instruction = ""
        if options.include_entities:
            entity_instruction = (
                " Include names of people, places, and important entities mentioned."
            )

        key_points_instruction = ""
        if options.preserve_key_points:
            key_points_instruction = " Preserve the most important points discussed."

        prompt = (
            "Please summarize the following conversation."
            f"{style_instructions}{length_instruction}{entity_instruction}{key_points_instruction} "
            "Focus on the main topics and key information exchanged."
        )

        return {"role": "system", "content": prompt}

    def _generate_cache_key(
        self, entries: List[ContextEntry], options: SummarizationOptions
    ) -> str:
        """
        Generate a cache key for a set of entries and options.

        Args:
            entries: The entries to summarize
            options: Summarization options

        Returns:
            A cache key string
        """
        # Use entry IDs and timestamps to create a unique key
        entry_ids = "-".join(
            [f"{entry.id[:8]}:{int(entry.timestamp)}" for entry in entries[:3]]
        )
        entry_count = len(entries)
        options_str = f"{options.style}:{options.max_length or 0}"
        return f"{entry_count}:{entry_ids}:{options_str}"

    def _create_fallback_summary(self, entries: List[ContextEntry]) -> ContextEntry:
        """
        Create a fallback summary when LLM generation fails.

        Args:
            entries: The entries to summarize

        Returns:
            A basic summary entry
        """
        # Count messages by role
        role_counts: Dict[str, int] = {}
        for entry in entries:
            role_counts[entry.role] = role_counts.get(entry.role, 0) + 1

        # Create a simple summary
        roles_str = ", ".join(
            [f"{count} {role}" for role, count in role_counts.items()]
        )
        first_timestamp = min(entries, key=lambda e: e.timestamp).timestamp
        last_timestamp = max(entries, key=lambda e: e.timestamp).timestamp
        time_span = int((last_timestamp - first_timestamp) / 60)  # minutes

        summary = (
            f"Previous conversation with {len(entries)} messages "
            f"({roles_str}) spanning {time_span} minutes. "
            "Content not available in this summary."
        )

        return ContextEntry(
            role="system",
            content=f"Previous conversation summary: {summary}",
            metadata={
                "is_summary": True,
                "is_fallback": True,
                "summarized_count": len(entries),
                "summarization_timestamp": time.time(),
            },
        )
