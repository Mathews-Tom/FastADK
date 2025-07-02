"""
Memory backends for FastADK.

This module provides the base memory interface and implementations for
different memory backends (in-memory, Redis, etc.)
"""

from typing import Optional

from fastadk.core.config import MemoryBackendType, get_settings

# Re-export for convenience
from .base import MemoryBackend, MemoryEntry
from .inmemory import InMemoryBackend

__all__ = ["MemoryBackend", "MemoryEntry", "InMemoryBackend", "get_memory_backend"]


def get_memory_backend(
    backend_type: MemoryBackendType | None = None,
) -> MemoryBackend:
    """
    Get a memory backend instance based on configuration.

    Args:
        backend_type: Optional override for the backend type from config

    Returns:
        A memory backend instance
    """
    settings = get_settings()

    # Use provided backend type or get from settings
    memory_type = backend_type or settings.memory.backend_type

    if memory_type == MemoryBackendType.IN_MEMORY:
        return InMemoryBackend()

    elif memory_type == MemoryBackendType.REDIS:
        try:
            from .redis import RedisBackend  # type: ignore

            return RedisBackend(  # type: ignore
                connection_string=settings.memory.connection_string,
                ttl_seconds=settings.memory.ttl_seconds,
                **settings.memory.options,
            )
        except ImportError as exc:
            raise ImportError(
                "Redis memory backend requires extra dependencies. "
                "Install them with: uv add fastadk[redis]"
            ) from exc

    elif memory_type == MemoryBackendType.FIRESTORE:
        try:
            from .firestore import FirestoreBackend  # type: ignore

            return FirestoreBackend(  # type: ignore
                connection_string=settings.memory.connection_string,
                ttl_seconds=settings.memory.ttl_seconds,
                **settings.memory.options,
            )
        except ImportError as exc:
            raise ImportError(
                "Firestore memory backend requires extra dependencies. "
                "Install them with: uv add fastadk[firestore]"
            ) from exc

    elif memory_type == MemoryBackendType.CUSTOM:
        # Custom backend handling
        # This would import from a user-specified module
        # For now, fall back to in-memory
        return InMemoryBackend()

    else:
        # Default to in-memory if type is unknown
        return InMemoryBackend()
