"""
Vector memory backend for FastADK.

This module provides a vector-based memory backend for semantic storage and retrieval.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from fastadk.memory.base import MemoryBackend, MemoryEntry

# Define a type for vector embeddings
Embedding = List[float]


class VectorEntry(BaseModel):
    """
    Vector entry containing both the data and its embedding.

    This extends the basic MemoryEntry with vector embedding capabilities.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    data: Any
    embedding: Optional[Embedding] = None
    created_at: float = Field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingProviderProtocol:
    """Protocol for embedding providers."""

    async def get_embedding(self, text: str) -> Embedding:
        """
        Get an embedding vector for a text string.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        raise NotImplementedError()


class VectorStoreProtocol:
    """Protocol for vector stores."""

    async def add(
        self, id: str, embedding: Embedding, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Optional metadata
        """
        raise NotImplementedError()

    async def search(
        self, query_embedding: Embedding, limit: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of tuples with (id, similarity_score)
        """
        raise NotImplementedError()

    async def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            id: Identifier of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError()

    async def clear(self) -> int:
        """
        Clear all vectors from the store.

        Returns:
            Number of vectors deleted
        """
        raise NotImplementedError()

    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        raise NotImplementedError()


class InMemoryVectorStore(VectorStoreProtocol):
    """
    Simple in-memory vector store for development and testing.

    This implementation stores vectors in memory and performs
    similarity search using cosine similarity.
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self.vectors: Dict[str, Tuple[Embedding, Dict[str, Any]]] = {}

    async def add(
        self, id: str, embedding: Embedding, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Optional metadata
        """
        self.vectors[id] = (embedding, metadata or {})

    async def search(
        self, query_embedding: Embedding, limit: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of tuples with (id, similarity_score)
        """
        if not self.vectors:
            return []

        # Calculate similarity scores for all vectors
        similarities = []
        for id, (embedding, _) in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((id, similarity))

        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    async def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            id: Identifier of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        if id in self.vectors:
            del self.vectors[id]
            return True
        return False

    async def clear(self) -> int:
        """
        Clear all vectors from the store.

        Returns:
            Number of vectors deleted
        """
        count = len(self.vectors)
        self.vectors.clear()
        return count

    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy.

        For the in-memory implementation, this always returns True.

        Returns:
            True if healthy, False otherwise
        """
        return True

    def _cosine_similarity(self, vec1: Embedding, vec2: Embedding) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        if len(vec1) != len(vec2):
            # If dimensions don't match, return 0 similarity
            return 0.0

        # Special case: if the vectors are identical, return 1.0
        if vec1 == vec2:
            return 1.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)
        # Ensure the result is between 0 and 1 (floating-point precision issues)
        similarity_float = float(max(0.0, min(1.0, similarity)))
        return similarity_float


class VectorMemoryBackend(MemoryBackend):
    """
    Vector-based memory backend for semantic storage and retrieval.

    This backend stores data along with vector embeddings, allowing for
    semantic search capabilities.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        embedding_provider: EmbeddingProviderProtocol,
        storage_backend: Optional[MemoryBackend] = None,
    ):
        """
        Initialize the vector memory backend.

        Args:
            vector_store: The vector store for embeddings
            embedding_provider: Service to generate embeddings
            storage_backend: Backend for storing the actual data (optional)
        """
        self.vector_store = vector_store
        self.embedder = embedding_provider
        self.storage = storage_backend

    async def get(self, key: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by key.

        Args:
            key: The key to look up

        Returns:
            The memory entry if found and not expired, otherwise None
        """
        # If we have a storage backend, delegate to it
        if self.storage:
            return await self.storage.get(key)

        # Otherwise we don't support direct key lookup
        # (only semantic search is supported)
        return None

    async def set(
        self, key: str, data: Any, ttl_seconds: Optional[int] = None
    ) -> MemoryEntry:
        """
        Store a value in memory with vector embedding.

        Args:
            key: The key to store the data under
            data: The data to store
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            The created memory entry
        """
        # Calculate expiration time if TTL is provided
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        # Convert data to string for embedding if it's not already
        text_to_embed = ""
        if isinstance(data, str):
            text_to_embed = data
        elif isinstance(data, dict):
            # For dictionaries, concatenate all string values
            for value in data.values():
                if isinstance(value, str):
                    text_to_embed += value + " "
        else:
            # For other types, use string representation
            text_to_embed = str(data)

        # Generate embedding
        try:
            embedding = await self.embedder.get_embedding(text_to_embed)
        except Exception as e:
            logger.error("Failed to generate embedding for key %s: %s", key, str(e))
            embedding = None

        # Create entry
        entry = MemoryEntry(
            key=key,
            data=data,
            created_at=time.time(),
            expires_at=expires_at,
        )

        # Store in vector store if we have an embedding
        if embedding:
            entry_id = f"memory:{key}"
            metadata = {
                "key": key,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
            }
            await self.vector_store.add(entry_id, embedding, metadata)

        # If we have a storage backend, also store there
        if self.storage:
            await self.storage.set(key, data, ttl_seconds)

        return entry

    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        deleted_from_vector = await self.vector_store.delete(f"memory:{key}")

        # If we have a storage backend, also delete from there
        if self.storage:
            deleted_from_storage = await self.storage.delete(key)
            return deleted_from_vector or deleted_from_storage

        return deleted_from_vector

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        # If we have a storage backend, delegate to it
        if self.storage:
            return await self.storage.exists(key)

        # Otherwise we don't support direct key existence check
        return False

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching a pattern.

        This operation is only supported if a storage backend is provided.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            List of matching keys
        """
        # If we have a storage backend, delegate to it
        if self.storage:
            return await self.storage.keys(pattern)

        # Otherwise we don't support listing keys
        return []

    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear entries from memory.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            Number of entries cleared
        """
        if not pattern:
            # Clear everything in vector store
            vector_count = await self.vector_store.clear()

            # If we have a storage backend, also clear that
            if self.storage:
                storage_count = await self.storage.clear()
                return max(vector_count, storage_count)

            return vector_count

        # For pattern-based clearing, we need a storage backend
        if self.storage:
            # Get keys matching the pattern
            keys = await self.storage.keys(pattern)
            count = 0

            # Delete each key from both stores
            for key in keys:
                deleted = await self.delete(key)
                if deleted:
                    count += 1

            return count

        # If no storage backend, we can't do pattern-based clearing
        return 0

    async def ttl(self, key: str) -> Optional[float]:
        """
        Get the time-to-live for a key in seconds.

        Args:
            key: The key to check

        Returns:
            The TTL in seconds, or None if the key doesn't exist or has no expiration
        """
        # If we have a storage backend, delegate to it
        if self.storage:
            return await self.storage.ttl(key)

        # Otherwise we don't support TTL queries
        return None

    async def search_semantic(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Search for semantically similar entries.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of matching memory entries
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedder.get_embedding(query)

            # Search for similar vectors
            results = await self.vector_store.search(
                query_embedding, limit=limit, threshold=threshold
            )

            # Collect the entries
            entries = []
            for id, similarity in results:
                # Extract key from id
                if id.startswith("memory:"):
                    key = id[7:]  # Remove "memory:" prefix
                else:
                    key = id

                # Get the entry data
                if self.storage:
                    entry = await self.storage.get(key)
                    if entry:
                        # Add similarity score to metadata
                        entry.metadata["similarity_score"] = similarity
                        entries.append(entry)
                else:
                    # Without a storage backend, we can only return basic entries
                    entries.append(
                        MemoryEntry(
                            key=key,
                            data=None,  # Data not available
                            created_at=time.time(),
                            expires_at=None,
                            metadata={"similarity_score": similarity},
                        )
                    )

            return entries
        except Exception as e:
            logger.error("Error in semantic search: %s", str(e))
            # Return empty list on error
            return []

    async def health_check(self) -> bool:
        """
        Check if the memory backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        # Check vector store health
        vector_healthy = await self.vector_store.health_check()

        # If we have a storage backend, also check its health
        if self.storage:
            storage_healthy = await self.storage.health_check()
            return vector_healthy and storage_healthy

        return vector_healthy


class MockEmbeddingProvider(EmbeddingProviderProtocol):
    """
    Mock embedding provider for testing.

    This provider generates deterministic but meaningless embeddings
    suitable for testing vector operations without requiring a real model.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize the mock embedding provider.

        Args:
            dimension: Dimension of the embeddings to generate
        """
        self.dimension = dimension

    async def get_embedding(self, text: str) -> Embedding:
        """
        Get a mock embedding for a text string.

        This implementation uses a simple hash of the text to generate
        pseudo-random but deterministic embeddings.

        Args:
            text: The text to embed

        Returns:
            A mock embedding vector
        """
        # Use hash of text to seed the embedding values
        import hashlib

        hash_obj = hashlib.sha256(text.encode(), usedforsecurity=False)
        hash_bytes = hash_obj.digest()

        # Generate embedding values between -1 and 1
        embedding = []
        for i in range(self.dimension):
            # Use different bytes from the hash as seeds
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)

        # Normalize the vector to unit length
        magnitude = sum(v * v for v in embedding) ** 0.5
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]

        return embedding
