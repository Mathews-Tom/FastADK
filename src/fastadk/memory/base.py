"""
Base memory backend classes for FastADK.

This module provides the base abstract classes for memory backends.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str = Field(..., description="Unique key for this memory entry")
    data: Any = Field(..., description="The data stored in this memory entry")
    created_at: float = Field(
        default_factory=time.time, description="Creation timestamp"
    )
    expires_at: float | None = Field(
        None, description="Expiration timestamp (None means no expiration)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this entry"
    )

    def is_expired(self) -> bool:
        """Check if this entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_ttl(self) -> float | None:
        """Get the remaining time-to-live in seconds."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)


class MemoryBackend(ABC):
    """
    Abstract base class for memory backends.

    All memory backends must implement these methods to provide
    consistent storage and retrieval functionality.
    """

    @abstractmethod
    async def get(self, key: str) -> MemoryEntry | None:
        """
        Get a memory entry by key.

        Args:
            key: The key to look up

        Returns:
            The memory entry if found, otherwise None
        """

    @abstractmethod
    async def set(
        self, key: str, data: Any, ttl_seconds: int | None = None
    ) -> MemoryEntry:
        """
        Store a value in memory.

        Args:
            key: The key to store the data under
            data: The data to store
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            The created memory entry
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """

    @abstractmethod
    async def keys(self, pattern: str | None = None) -> list[str]:
        """
        Get all keys matching a pattern.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            List of matching keys
        """

    @abstractmethod
    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear entries from memory.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            Number of entries cleared
        """

    @abstractmethod
    async def ttl(self, key: str) -> float | None:
        """
        Get the time-to-live for a key in seconds.

        Args:
            key: The key to check

        Returns:
            The TTL in seconds, or None if the key doesn't exist or has no expiration
        """

    @abstractmethod
    async def search_semantic(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> list[MemoryEntry]:
        """
        Search for semantically similar entries.

        This is a placeholder for vector-based backends.
        Basic backends can implement a simple search.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of matching memory entries
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the memory backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
