"""
Tests for vector memory backend.

This module tests the vector memory backend implementation.
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fastadk.memory.inmemory import InMemoryBackend
from fastadk.memory.vector import (
    InMemoryVectorStore,
    MockEmbeddingProvider,
    VectorMemoryBackend,
)


class TestMockEmbeddingProvider:
    """Tests for the MockEmbeddingProvider class."""

    @pytest.mark.asyncio
    async def test_get_embedding(self):
        """Test getting an embedding from the mock provider."""
        provider = MockEmbeddingProvider(dimension=10)

        # Get an embedding
        embedding = await provider.get_embedding("test text")

        # Verify properties
        assert len(embedding) == 10
        assert all(isinstance(value, float) for value in embedding)
        assert all(-1 <= value <= 1 for value in embedding)

        # Verify deterministic behavior (same input = same embedding)
        embedding2 = await provider.get_embedding("test text")
        assert embedding == embedding2

        # Verify different inputs produce different embeddings
        embedding3 = await provider.get_embedding("different text")
        assert embedding != embedding3


class TestInMemoryVectorStore:
    """Tests for the InMemoryVectorStore class."""

    @pytest_asyncio.fixture
    async def vector_store(self):
        """Create a vector store for testing."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_add_and_search(self, vector_store):
        """Test adding vectors and searching."""
        # Add some vectors
        await vector_store.add("id1", [0.1, 0.2, 0.3])
        await vector_store.add("id2", [0.2, 0.3, 0.4])
        await vector_store.add("id3", [0.9, 0.8, 0.7])

        # Search for similar vectors
        results = await vector_store.search([0.1, 0.2, 0.3], limit=2)

        # Verify results
        assert len(results) == 2
        # First result should be the exact match (id1)
        assert results[0][0] == "id1"
        assert results[0][1] > 0.99  # Should be very close to 1.0
        # Second result should be id2 (more similar than id3)
        assert results[1][0] == "id2"

    @pytest.mark.asyncio
    async def test_search_with_exact_match(self, vector_store):
        """Test searching for an exact match."""
        # Add some vectors
        test_vector = [0.1, 0.2, 0.3]
        await vector_store.add("id1", test_vector)
        await vector_store.add("id2", [0.2, 0.3, 0.4])
        await vector_store.add("id3", [0.9, 0.8, 0.7])

        # Search for exact match
        results = await vector_store.search(test_vector)

        # The exact match should be the first result with similarity 1.0
        assert results[0][0] == "id1"
        assert abs(results[0][1] - 1.0) < 1e-6  # Should be very close to 1.0

    @pytest.mark.asyncio
    async def test_delete(self, vector_store):
        """Test deleting vectors."""
        # Add a vector
        await vector_store.add("id1", [0.1, 0.2, 0.3])

        # Delete it
        result = await vector_store.delete("id1")
        assert result is True

        # Try to delete a non-existent vector
        result = await vector_store.delete("non-existent")
        assert result is False

        # Verify it's gone from search results
        results = await vector_store.search([0.1, 0.2, 0.3])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_clear(self, vector_store):
        """Test clearing all vectors."""
        # Add some vectors
        await vector_store.add("id1", [0.1, 0.2, 0.3])
        await vector_store.add("id2", [0.2, 0.3, 0.4])

        # Clear all
        count = await vector_store.clear()
        assert count == 2

        # Verify all are gone
        results = await vector_store.search([0.1, 0.2, 0.3])
        assert len(results) == 0


class TestVectorMemoryBackend:
    """Tests for the VectorMemoryBackend class."""

    @pytest_asyncio.fixture
    async def vector_store(self):
        """Create a vector store for testing."""
        return InMemoryVectorStore()

    @pytest_asyncio.fixture
    async def embedding_provider(self):
        """Create an embedding provider for testing."""
        return MockEmbeddingProvider()

    @pytest_asyncio.fixture
    async def storage_backend(self):
        """Create a storage backend for testing."""
        return InMemoryBackend()

    @pytest_asyncio.fixture
    async def vector_memory(self, vector_store, embedding_provider, storage_backend):
        """Create a vector memory backend for testing."""
        return VectorMemoryBackend(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            storage_backend=storage_backend,
        )

    @pytest.mark.asyncio
    async def test_set_and_search(self, vector_memory, monkeypatch):
        """Test setting values and searching."""

        # Mock the embedding provider to return deterministic embeddings
        async def mock_get_embedding(text):
            if "Python" in text:
                return [0.9, 0.1, 0.1]  # Python-related embedding
            elif "JavaScript" in text:
                return [0.1, 0.9, 0.1]  # JavaScript-related embedding
            else:
                return [0.1, 0.1, 0.9]  # Other embedding

        monkeypatch.setattr(vector_memory.embedder, "get_embedding", mock_get_embedding)

        # Set some values
        await vector_memory.set("key1", "This is a test about Python")
        await vector_memory.set("key2", "This is about JavaScript")
        await vector_memory.set("key3", "Nothing to do with programming")

        # Search with Python-like embedding
        results = await vector_memory.search_semantic("Python programming")

        # Verify results
        assert len(results) > 0
        # Python should be found since we're using deterministic embeddings
        assert any(
            "Python" in entry.data for entry in results if entry.data is not None
        )

    @pytest.mark.asyncio
    async def test_get_and_delete(self, vector_memory):
        """Test getting and deleting entries."""
        # With storage backend
        # Set a value
        await vector_memory.set("key1", "test value")

        # Get it
        entry = await vector_memory.get("key1")
        assert entry is not None
        assert entry.key == "key1"
        assert entry.data == "test value"

        # Delete it
        result = await vector_memory.delete("key1")
        assert result is True

        # Get it again (should be None)
        entry = await vector_memory.get("key1")
        assert entry is None

    @pytest.mark.asyncio
    async def test_without_storage_backend(self, vector_store, embedding_provider):
        """Test operation without a storage backend."""
        # Create backend without storage
        vector_memory = VectorMemoryBackend(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

        # Set a value
        await vector_memory.set("key1", "test value")

        # Get it (should be None without storage)
        entry = await vector_memory.get("key1")
        assert entry is None

        # But we should still be able to search
        results = await vector_memory.search_semantic("test")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_health_check(self, vector_memory, vector_store, storage_backend):
        """Test health check."""
        # Mock health checks
        vector_store.health_check = AsyncMock(return_value=True)
        storage_backend.health_check = AsyncMock(return_value=True)

        # Check health
        result = await vector_memory.health_check()
        assert result is True

        # Verify both were called
        vector_store.health_check.assert_called_once()
        storage_backend.health_check.assert_called_once()

        # Test unhealthy vector store
        vector_store.health_check = AsyncMock(return_value=False)
        result = await vector_memory.health_check()
        assert result is False
