"""
Context management for FastADK.

This module provides context management capabilities for agent conversations,
including long-term memory, summarization, and context window management.
"""

import hashlib
import time
import uuid
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from fastadk.core.context_policy import ContextPolicy, MostRecentPolicy
from fastadk.core.exceptions import OperationError
from fastadk.memory.base import MemoryBackend


class ContextEntry(BaseModel):
    """A single entry in the conversation context."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this entry",
    )
    role: str = Field(
        ..., description="Role of the speaker (user, assistant, system, tool)"
    )
    content: str = Field(..., description="The content of the message")
    timestamp: float = Field(
        default_factory=time.time, description="When this message was created"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ContextWindow(BaseModel):
    """
    A sliding window of conversation context.

    This represents the recent conversation history that will be included
    in the prompt sent to the LLM.
    """

    entries: list[ContextEntry] = Field(
        default_factory=list, description="Messages in this context window"
    )
    max_entries: int = Field(10, description="Maximum number of entries to keep")
    max_tokens: int | None = Field(
        None, description="Maximum tokens to include in window"
    )
    summary: str | None = Field(
        default=None, description="Summary of older context not in window"
    )

    def add_entry(self, entry: ContextEntry) -> None:
        """
        Add an entry to the context window.

        Args:
            entry: The entry to add
        """
        self.entries.append(entry)
        self._trim_to_size()

    def _trim_to_size(self) -> None:
        """Trim the context window to fit within the configured limits."""
        # First trim by entry count
        max_entries = int(self.max_entries)  # Ensure it's an integer
        if max_entries and len(self.entries) > max_entries:
            # Keep only the most recent entries by removing from the beginning
            excess = len(self.entries) - max_entries
            self.entries = self.entries[excess:]

        # TODO: Implement token-based trimming when max_tokens is set
        # This would require a tokenizer integration

    def get_entries(self, roles: list[str] | None = None) -> list[ContextEntry]:
        """
        Get entries filtered by role.

        Args:
            roles: Optional list of roles to include

        Returns:
            List of entries matching the role filter
        """
        if roles is None:
            return self.entries.copy()
        return [entry for entry in self.entries if entry.role in roles]


class ConversationContext:
    """
    Manages conversation context for an agent.

    This class provides methods for:
    - Tracking conversation history
    - Persisting to memory backend
    - Summarizing past conversation
    - Managing context window size
    - Integrating long and short term memory
    """

    def __init__(
        self,
        session_id: str,
        memory_backend: MemoryBackend,
        context_policy: Optional[ContextPolicy] = None,
        window_size: int = 10,
        max_tokens: int | None = None,
        summarize_threshold: int = 20,
    ):
        """
        Initialize conversation context.

        Args:
            session_id: Unique identifier for this conversation
            memory_backend: Backend for storing conversation history
            context_policy: Policy for managing context window (optional)
            window_size: Maximum number of entries in context window
            max_tokens: Maximum tokens in context window (if supported)
            summarize_threshold: When to trigger summarization (number of entries)
        """
        self.session_id = session_id
        self.memory = memory_backend
        self.window = ContextWindow(max_entries=window_size, max_tokens=max_tokens)
        self.summarize_threshold = summarize_threshold
        self._is_loaded = False
        self._needs_summarization = False
        self._full_history: List[ContextEntry] = []

        # Set the context policy (default to MostRecentPolicy if not provided)
        self.context_policy = context_policy or MostRecentPolicy(
            max_messages=window_size
        )

    async def load(self) -> None:
        """
        Load context from memory backend.

        This populates the context window and full history from the memory backend.
        """
        if self._is_loaded:
            return

        # Load context window
        window_key = f"context:{self.session_id}:window"
        window_entry = await self.memory.get(window_key)

        if window_entry and window_entry.data:
            window_data = window_entry.data
            self.window = ContextWindow(**window_data)

        # Load full history
        history_key = f"context:{self.session_id}:history"
        history_entry = await self.memory.get(history_key)

        if history_entry and history_entry.data:
            self._full_history = [ContextEntry(**entry) for entry in history_entry.data]

        self._is_loaded = True
        logger.debug("Loaded context for session %s", self.session_id)

    async def save(self) -> None:
        """
        Save context to memory backend.

        This persists the context window and full history to the memory backend.
        """
        if not self._is_loaded:
            await self.load()

        # Save context window
        window_key = f"context:{self.session_id}:window"
        window_data = self.window.model_dump()
        await self.memory.set(window_key, window_data)

        # Save full history
        history_key = f"context:{self.session_id}:history"
        history_data = [entry.model_dump() for entry in self._full_history]
        await self.memory.set(history_key, history_data)

        logger.debug("Saved context for session %s", self.session_id)

    async def add_entry(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> ContextEntry:
        """
        Add an entry to the conversation context.

        Args:
            role: Role of the speaker (user, assistant, system, tool)
            content: Content of the message
            metadata: Optional additional metadata

        Returns:
            The created context entry
        """
        if not self._is_loaded:
            await self.load()

        entry = ContextEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        # Add to both window and full history
        self.window.add_entry(entry)
        self._full_history.append(entry)

        # Check if we need to summarize
        if (
            self.summarize_threshold > 0
            and len(self._full_history) >= self.summarize_threshold
            and len(self._full_history) % self.summarize_threshold == 0
        ):
            self._needs_summarization = True

        # Save changes
        await self.save()
        return entry

    async def get_entries(
        self, roles: list[str] | None = None, full_history: bool = False
    ) -> list[ContextEntry]:
        """
        Get context entries, optionally filtered by role.

        Args:
            roles: Optional list of roles to include
            full_history: Whether to return the full history or just the window

        Returns:
            List of context entries
        """
        if not self._is_loaded:
            await self.load()

        if full_history:
            entries = self._full_history
        else:
            entries = self.window.entries

        if roles:
            return [entry for entry in entries if entry.role in roles]
        return entries.copy()

    async def get_formatted_context(
        self, include_summary: bool = True
    ) -> list[dict[str, str]]:
        """
        Get the context in a format suitable for LLM prompting.

        Args:
            include_summary: Whether to include context summary

        Returns:
            List of formatted messages
        """
        if not self._is_loaded:
            await self.load()

        # Check if we need to generate a summary first
        if self._needs_summarization:
            await self.summarize_older_context()

        # Apply context policy to determine which messages to include
        if self.context_policy:
            context_entries = await self.context_policy.apply(self._full_history)
        else:
            context_entries = self.window.entries

        formatted = []

        # Add summary if available and requested
        if include_summary and self.window.summary:
            formatted.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {self.window.summary}",
                }
            )

        # Add entries from the context policy
        for entry in context_entries:
            formatted.append({"role": entry.role, "content": entry.content})

        return formatted

    async def clear(self) -> None:
        """Clear all context data for this session."""
        window_key = f"context:{self.session_id}:window"
        history_key = f"context:{self.session_id}:history"
        summary_key = f"context:{self.session_id}:summary"

        await self.memory.delete(window_key)
        await self.memory.delete(history_key)
        await self.memory.delete(summary_key)

        self.window = ContextWindow(
            max_entries=self.window.max_entries, max_tokens=self.window.max_tokens
        )
        self._full_history = []
        self._is_loaded = True
        self._needs_summarization = False

        logger.debug("Cleared context for session %s", self.session_id)

    async def summarize_older_context(self) -> str:
        """
        Summarize older context entries not in the current window.

        This method uses an LLM to generate a summary of older context
        that's no longer in the active window.

        Returns:
            The generated summary
        """
        if not self._is_loaded:
            await self.load()

        # If we don't have enough history to summarize, skip
        if len(self._full_history) <= self.window.max_entries:
            self._needs_summarization = False
            return ""

        # Get entries that are in full history but not in window
        window_ids = {entry.id for entry in self.window.entries}
        entries_to_summarize = [
            entry for entry in self._full_history if entry.id not in window_ids
        ]

        if not entries_to_summarize:
            self._needs_summarization = False
            return ""

        # TODO: Implement actual LLM summarization
        # For now, we'll just create a simple summary
        roles = {entry.role for entry in entries_to_summarize}
        summary = (
            f"Previous conversation with {len(entries_to_summarize)} messages "
            f"between {', '.join(roles)}."
        )

        # Store the summary
        self.window.summary = summary
        await self.save()

        self._needs_summarization = False
        return summary

    async def search_semantic(
        self, query: str, limit: int = 5, threshold: float = 0.0
    ) -> list[ContextEntry]:
        """
        Search for semantically similar entries in the conversation history.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of matching context entries
        """
        if not self._is_loaded:
            await self.load()

        # For now, implement a simple keyword search
        # In the future, this would use vector embeddings for semantic search
        query_lower = query.lower()
        matches = []

        for entry in self._full_history:
            # Calculate a simple similarity score (0-1) based on substring match
            if query_lower in entry.content.lower():
                # Simple similarity score based on relative length of query vs content
                similarity = len(query_lower) / max(len(entry.content.lower()), 1)
                # Only include if above threshold
                if similarity >= threshold:
                    matches.append(entry)
                    if len(matches) >= limit:
                        break

        return matches

    def get_session_id(self) -> str:
        """Get the session ID for this context."""
        return self.session_id

    @staticmethod
    def generate_session_id(user_id: str = "") -> str:
        """
        Generate a unique session ID.

        Args:
            user_id: Optional user identifier to include in the session ID

        Returns:
            A unique session ID
        """
        timestamp = int(time.time())
        random_component = uuid.uuid4().hex[:8]

        if user_id:
            # Create a deterministic component based on user_id
            # Use SHA-256 which is more secure than MD5
            user_hash = hashlib.sha256(
                user_id.encode(), usedforsecurity=False
            ).hexdigest()[:8]
            return f"session_{user_hash}_{timestamp}_{random_component}"

        return f"session_{timestamp}_{random_component}"


class ContextManager:
    """
    Manages conversation contexts for multiple sessions.

    This class provides a centralized way to access and manage
    conversation contexts across different sessions.
    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        default_context_policy: Optional[ContextPolicy] = None,
    ):
        """
        Initialize the context manager.

        Args:
            memory_backend: Backend for storing conversation contexts
            default_context_policy: Default policy to apply to new contexts
        """
        self.memory = memory_backend
        self.default_policy = default_context_policy
        self._contexts: dict[str, ConversationContext] = {}

    async def get_context(
        self,
        session_id: str,
        create_if_missing: bool = True,
        context_policy: Optional[ContextPolicy] = None,
    ) -> ConversationContext:
        """
        Get a conversation context by session ID.

        Args:
            session_id: The session ID
            create_if_missing: Whether to create a new context if not found
            context_policy: Optional policy to apply to this context

        Returns:
            The conversation context

        Raises:
            OperationError: If the context doesn't exist and create_if_missing is False
        """
        if session_id in self._contexts:
            return self._contexts[session_id]

        # Check if context exists in memory
        exists = await self.memory.exists(f"context:{session_id}:window")

        if not exists and not create_if_missing:
            raise OperationError(
                message=f"Context for session {session_id} not found",
                error_code="CONTEXT_NOT_FOUND",
                details={"session_id": session_id},
            )

        # Use provided policy, default policy, or null policy in that order
        policy = context_policy or self.default_policy

        # Create a new context
        context = ConversationContext(
            session_id=session_id,
            memory_backend=self.memory,
            context_policy=policy,
        )

        # Initialize context
        await context.load()

        # Cache the context
        self._contexts[session_id] = context
        return context

    async def list_sessions(self, limit: int = 100) -> list[str]:
        """
        List active session IDs.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session IDs
        """
        # Get all keys matching the context pattern
        keys = await self.memory.keys("context:*:window")

        # Extract session IDs from keys
        session_ids = []
        for key in keys:
            if key.startswith("context:") and key.endswith(":window"):
                session_id = key.split(":")[1]
                session_ids.append(session_id)

        return session_ids[:limit]

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its context.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Check if the session exists
        exists = await self.memory.exists(f"context:{session_id}:window")
        if not exists:
            return False

        # Get the context and clear it
        context = await self.get_context(session_id, create_if_missing=False)
        await context.clear()

        # Remove from cache
        if session_id in self._contexts:
            del self._contexts[session_id]

        return True

    async def create_session(
        self,
        user_id: str = "",
        system_message: Optional[str] = None,
        context_policy: Optional[ContextPolicy] = None,
    ) -> ConversationContext:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            system_message: Optional system message to initialize the context
            context_policy: Optional policy to apply to this context

        Returns:
            The new conversation context
        """
        session_id = ConversationContext.generate_session_id(user_id)
        context = await self.get_context(
            session_id,
            create_if_missing=True,
            context_policy=context_policy,
        )

        # Add system message if provided
        if system_message:
            await context.add_entry(role="system", content=system_message)

        return context
