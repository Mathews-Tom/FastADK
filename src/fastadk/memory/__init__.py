"""
Memory backends for FastADK.

This module provides the base memory interface and implementations for
different memory backends (in-memory, Redis, vector, etc.)
"""

from typing import Optional

from fastadk.core.config import MemoryBackendType, get_settings

# Re-export for convenience
from .base import MemoryBackend, MemoryEntry
from .inmemory import InMemoryBackend

# Conditionally import backends to avoid ImportError
try:
    from .redis import RedisBackend
except ImportError:
    RedisBackend = None  # type: ignore

try:
    from .vector import (
        InMemoryVectorStore,
        MockEmbeddingProvider,
        VectorEntry,
        VectorMemoryBackend,
    )
except ImportError:
    VectorMemoryBackend = None  # type: ignore
    VectorEntry = None  # type: ignore
    InMemoryVectorStore = None  # type: ignore
    MockEmbeddingProvider = None  # type: ignore

__all__ = [
    "MemoryBackend",
    "MemoryEntry",
    "InMemoryBackend",
    "VectorMemoryBackend",
    "VectorEntry",
    "InMemoryVectorStore",
    "MockEmbeddingProvider",
    "get_memory_backend",
]


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
        if RedisBackend is None:
            raise ImportError(
                "Redis memory backend requires extra dependencies. "
                "Install them with: uv add fastadk[redis]"
            )

        # Extract Redis connection options from settings
        redis_options = settings.memory.options.copy()

        # Check for connection string in format: redis://user:password@host:port/db
        connection_string = settings.memory.connection_string
        if connection_string and connection_string.startswith("redis://"):
            # Connection string parsing would go here
            # For now, we'll use the options directly
            pass

        # Set default TTL if specified in settings
        if settings.memory.ttl_seconds:
            redis_options.setdefault("default_ttl", settings.memory.ttl_seconds)

        return RedisBackend(**redis_options)

    elif memory_type == MemoryBackendType.VECTOR:
        if VectorMemoryBackend is None:
            raise ImportError(
                "Vector memory backend requires extra dependencies. "
                "Install them with: uv add fastadk[vector]"
            )

        # For development/testing, use the mock embedding provider
        # In production, users would configure a real embedding provider
        vector_options = settings.memory.options.copy()
        use_mock = vector_options.pop("use_mock_embeddings", True)

        if use_mock:
            # Use the mock provider for testing
            vector_store = InMemoryVectorStore()
            embedding_provider = MockEmbeddingProvider()
        else:
            # Import real providers if specified in options
            embedding_provider_path = vector_options.pop("embedding_provider", None)
            vector_store_path = vector_options.pop("vector_store", None)

            if embedding_provider_path and vector_store_path:
                # Dynamic import of user-provided implementations
                import importlib

                try:
                    module_path, class_name = embedding_provider_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    EmbeddingProviderClass = getattr(module, class_name)
                    embedding_provider = EmbeddingProviderClass(
                        **vector_options.get("embedding_options", {})
                    )

                    module_path, class_name = vector_store_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    VectorStoreClass = getattr(module, class_name)
                    vector_store = VectorStoreClass(
                        **vector_options.get("store_options", {})
                    )
                except (ImportError, AttributeError) as e:
                    raise ImportError(
                        f"Failed to import vector components: {str(e)}"
                    ) from e
            else:
                # Fall back to mock provider
                vector_store = InMemoryVectorStore()
                embedding_provider = MockEmbeddingProvider()

        # Use another memory backend for storage if specified
        storage_backend = None
        storage_type = vector_options.pop("storage_backend_type", None)
        if storage_type:
            storage_backend = get_memory_backend(storage_type)

        return VectorMemoryBackend(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            storage_backend=storage_backend,
        )

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
