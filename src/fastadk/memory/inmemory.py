"""
In-memory backend for FastADK.

This module provides a simple in-memory backend for development and testing.
"""

import fnmatch
import time
from typing import Any

from .base import MemoryBackend, MemoryEntry


class InMemoryBackend(MemoryBackend):
    """
    In-memory implementation of the memory backend.

    This backend stores all data in memory, which makes it suitable for
    development and testing, but not for production use as data will be
    lost when the process restarts.
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._data: dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> MemoryEntry | None:
        """
        Get a memory entry by key.

        Args:
            key: The key to look up

        Returns:
            The memory entry if found and not expired, otherwise None
        """
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            # Clean up expired entry
            await self.delete(key)
            return None
        return entry

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
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds
        entry = MemoryEntry(
            key=key,
            data=data,
            expires_at=expires_at,
            created_at=time.time(),
        )
        self._data[key] = entry
        return entry

    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory and is not expired.

        Args:
            key: The key to check

        Returns:
            True if the key exists and is not expired, False otherwise
        """
        entry = await self.get(key)
        return entry is not None

    async def keys(self, pattern: str | None = None) -> list[str]:
        """
        Get all keys matching a pattern.

        Args:
            pattern: Optional glob pattern to match keys against

        Returns:
            List of matching keys
        """
        # Remove expired entries first
        await self._clean_expired()
        if pattern is None:
            return list(self._data.keys())
        # Use fnmatch for simple glob-style pattern matching
        return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]

    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear entries from memory.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            Number of entries cleared
        """
        if pattern is None:
            count = len(self._data)
            self._data.clear()
            return count
        keys_to_delete = await self.keys(pattern)
        for key in keys_to_delete:
            await self.delete(key)
        return len(keys_to_delete)

    async def ttl(self, key: str) -> float | None:
        """
        Get the time-to-live for a key in seconds.

        Args:
            key: The key to check

        Returns:
            The TTL in seconds, or None if the key doesn't exist or has no expiration
        """
        entry = await self.get(key)
        if entry is None:
            return None
        return entry.get_ttl()

    async def search_semantic(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> list[MemoryEntry]:
        """
        Search for entries that contain the query.

        This is a simple implementation that just checks if the query string
        is contained in the data. For production use, use a proper vector backend.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (unused in this implementation)

        Returns:
            List of matching memory entries
        """
        # Remove expired entries first
        await self._clean_expired()
        results = []
        query_lower = query.lower()
        for entry in self._data.values():
            # Simple contains check for string data
            if isinstance(entry.data, str) and query_lower in entry.data.lower():
                results.append(entry)
            # For dictionaries, check if any string value contains the query
            elif isinstance(entry.data, dict):
                for value in entry.data.values():
                    if isinstance(value, str) and query_lower in value.lower():
                        results.append(entry)
                        break
            if len(results) >= limit:
                break
        return results[:limit]

    async def health_check(self) -> bool:
        """
        Check if the memory backend is healthy.

        For the in-memory backend, this always returns True.

        Returns:
            True if healthy, False otherwise
        """
        return True

    async def _clean_expired(self) -> int:
        """
        Clean up expired entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._data.items()
            if entry.expires_at is not None and entry.expires_at < current_time
        ]
        for key in expired_keys:
            del self._data[key]
        return len(expired_keys)
