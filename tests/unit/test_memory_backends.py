"""
Tests for memory backends.

This module tests the different memory backends (InMemory, Redis, etc.)
"""

import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fastadk.memory.inmemory import InMemoryBackend


class TestInMemoryBackend:
    """Tests for the InMemoryBackend class."""

    @pytest_asyncio.fixture
    async def memory(self):
        """Create a new InMemoryBackend instance."""
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_set_get(self, memory):
        """Test setting and getting a value."""
        # Set a value
        await memory.set("test-key", "test-value")

        # Get the value
        entry = await memory.get("test-key")
        assert entry is not None
        assert entry.key == "test-key"
        assert entry.data == "test-value"
        assert entry.expires_at is None

    @pytest.mark.asyncio
    async def test_set_get_with_ttl(self, memory):
        """Test setting and getting a value with TTL."""
        # Set a value with 1 second TTL
        await memory.set("test-key", "test-value", ttl_seconds=1)

        # Get the value immediately (should exist)
        entry = await memory.get("test-key")
        assert entry is not None
        assert entry.key == "test-key"
        assert entry.data == "test-value"
        assert entry.expires_at is not None

        # Wait for it to expire
        time.sleep(1.1)

        # Get the value again (should be None)
        entry = await memory.get("test-key")
        assert entry is None

    @pytest.mark.asyncio
    async def test_delete(self, memory):
        """Test deleting a value."""
        # Set a value
        await memory.set("test-key", "test-value")

        # Delete the value
        result = await memory.delete("test-key")
        assert result is True

        # Get the value (should be None)
        entry = await memory.get("test-key")
        assert entry is None

        # Delete a non-existent value
        result = await memory.delete("non-existent-key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, memory):
        """Test checking if a key exists."""
        # Set a value
        await memory.set("test-key", "test-value")

        # Check if the key exists
        exists = await memory.exists("test-key")
        assert exists is True

        # Check if a non-existent key exists
        exists = await memory.exists("non-existent-key")
        assert exists is False

    @pytest.mark.asyncio
    async def test_keys(self, memory):
        """Test getting all keys."""
        # Set some values
        await memory.set("test-key-1", "test-value-1")
        await memory.set("test-key-2", "test-value-2")
        await memory.set("other-key-1", "other-value-1")

        # Get all keys
        keys = await memory.keys()
        assert len(keys) == 3
        assert "test-key-1" in keys
        assert "test-key-2" in keys
        assert "other-key-1" in keys

        # Get keys matching a pattern
        keys = await memory.keys("test-*")
        assert len(keys) == 2
        assert "test-key-1" in keys
        assert "test-key-2" in keys
        assert "other-key-1" not in keys

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """Test clearing all keys."""
        # Set some values
        await memory.set("test-key-1", "test-value-1")
        await memory.set("test-key-2", "test-value-2")
        await memory.set("other-key-1", "other-value-1")

        # Clear all keys
        count = await memory.clear()
        assert count == 3

        # Get all keys (should be empty)
        keys = await memory.keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_clear_pattern(self, memory):
        """Test clearing keys matching a pattern."""
        # Set some values
        await memory.set("test-key-1", "test-value-1")
        await memory.set("test-key-2", "test-value-2")
        await memory.set("other-key-1", "other-value-1")

        # Clear keys matching a pattern
        count = await memory.clear("test-*")
        assert count == 2

        # Get all keys
        keys = await memory.keys()
        assert len(keys) == 1
        assert "other-key-1" in keys

    @pytest.mark.asyncio
    async def test_ttl(self, memory):
        """Test getting TTL for a key."""
        # Set a value with 5 second TTL
        await memory.set("test-key", "test-value", ttl_seconds=5)

        # Get the TTL
        ttl = await memory.ttl("test-key")
        assert ttl is not None
        assert 0 < ttl <= 5

        # Set a value without TTL
        await memory.set("test-key-2", "test-value-2")

        # Get the TTL
        ttl = await memory.ttl("test-key-2")
        assert ttl is None

    @pytest.mark.asyncio
    async def test_search_semantic(self, memory):
        """Test searching for semantically similar entries."""
        # Set some values
        await memory.set("test-key-1", "This is a test message")
        await memory.set("test-key-2", "Another test with different content")
        await memory.set("test-key-3", "Something completely different")

        # Search for "test"
        results = await memory.search_semantic("test")
        assert len(results) == 2
        assert any(entry.key == "test-key-1" for entry in results)
        assert any(entry.key == "test-key-2" for entry in results)

        # Search for "different"
        results = await memory.search_semantic("different")
        assert len(results) == 2
        assert any(entry.key == "test-key-2" for entry in results)
        assert any(entry.key == "test-key-3" for entry in results)

    @pytest.mark.asyncio
    async def test_health_check(self, memory):
        """Test health check."""
        # In-memory backend should always be healthy
        assert await memory.health_check() is True


# We'll use pytest's monkeypatch to mock Redis for unit testing
@pytest.mark.asyncio
async def test_redis_backend_initialization():
    """Test initializing RedisBackend with different parameters."""
    # Import here to avoid ImportError if redis package is not installed
    try:
        import importlib.util

        if importlib.util.find_spec("redis") is None:
            pytest.skip("Redis dependencies not installed")

        from fastadk.memory.redis import RedisBackend

        # Only run the test if redis is installed
        with patch("redis.asyncio.Redis") as mock_redis:
            # Setup the mock
            mock_instance = AsyncMock()
            mock_redis.return_value = mock_instance

            # Test with default parameters
            backend = RedisBackend()
            assert backend.prefix == "fastadk:"
            mock_redis.assert_called_with(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                password=None,
            )

            # Test with custom parameters
            backend = RedisBackend(
                host="redis-server",
                port=6380,
                db=1,
                password="secret",
                prefix="custom:",
                ssl=True,
            )
            assert backend.prefix == "custom:"
            mock_redis.assert_called_with(
                host="redis-server",
                port=6380,
                db=1,
                decode_responses=True,
                password="secret",
                ssl=True,
            )
    except ImportError:
        pytest.skip("Redis dependencies not installed")

    # Test with default parameters
    backend = RedisBackend()
    assert backend.prefix == "fastadk:"
    mock_redis.assert_called_with(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True,
        password=None,
    )

    # Test with custom parameters
    backend = RedisBackend(
        host="redis-server",
        port=6380,
        db=1,
        password="secret",
        prefix="custom:",
        ssl=True,
    )
    assert backend.prefix == "custom:"
    mock_redis.assert_called_with(
        host="redis-server",
        port=6380,
        db=1,
        decode_responses=True,
        password="secret",
        ssl=True,
    )


@pytest.mark.asyncio
async def test_redis_backend_operations():
    """Test Redis backend operations."""
    # Import here to avoid ImportError if redis package is not installed
    try:
        import importlib.util

        if importlib.util.find_spec("redis") is None:
            pytest.skip("Redis dependencies not installed")

        from fastadk.memory.redis import RedisBackend

        # Create a mock Redis client
        mock_redis = AsyncMock()

        # Mock the Redis initialization to avoid actual connection
        with patch("redis.asyncio.Redis", return_value=mock_redis):
            # Create the backend with the mock client
            backend = RedisBackend(host="localhost")

            # Test get operation
            mock_redis.get.return_value = '{"key": "test-key", "data": "test-value", "created_at": 1625097600.0, "expires_at": null, "metadata": {}}'
            entry = await backend.get("test-key")
            mock_redis.get.assert_called_with("fastadk:test-key")
            assert entry is not None
            assert entry.key == "test-key"
            assert entry.data == "test-value"

            # Test get with missing key
            mock_redis.get.return_value = None
            entry = await backend.get("missing-key")
            assert entry is None

            # Test set operation
            entry = await backend.set("test-key", "test-value")
            assert entry.key == "test-key"
            assert entry.data == "test-value"
            mock_redis.set.assert_called_once()

            # Test set with TTL
            mock_redis.reset_mock()
            entry = await backend.set("test-key", "test-value", ttl_seconds=60)
            assert entry.key == "test-key"
            assert entry.data == "test-value"
            assert entry.expires_at is not None
            mock_redis.setex.assert_called_once()

            # Test delete operation
            mock_redis.reset_mock()
            mock_redis.delete.return_value = 1
            result = await backend.delete("test-key")
            mock_redis.delete.assert_called_with("fastadk:test-key")
            assert result is True

            # Test delete non-existent key
            mock_redis.reset_mock()
            mock_redis.delete.return_value = 0
            result = await backend.delete("missing-key")
            assert result is False

            # Test exists operation
            mock_redis.reset_mock()
            mock_redis.exists.return_value = 1
            mock_redis.get.return_value = '{"key": "test-key", "data": "test-value", "created_at": 1625097600.0, "expires_at": null, "metadata": {}}'
            result = await backend.exists("test-key")
            mock_redis.exists.assert_called_with("fastadk:test-key")
            assert result is True

            # Test keys operation
            mock_redis.reset_mock()
            mock_redis.keys.return_value = ["fastadk:test-key-1", "fastadk:test-key-2"]
            keys = await backend.keys("test-*")
            mock_redis.keys.assert_called_with("fastadk:test-*")
            assert keys == ["test-key-1", "test-key-2"]

            # Test clear operation
            mock_redis.reset_mock()
            mock_redis.keys.return_value = ["fastadk:test-key-1", "fastadk:test-key-2"]
            mock_redis.delete.return_value = 2
            count = await backend.clear("test-*")
            assert count == 2

            # Test health check
            mock_redis.reset_mock()
            mock_redis.ping.return_value = True
            result = await backend.health_check()
            mock_redis.ping.assert_called_once()
            assert result is True
    except ImportError:
        pytest.skip("Redis dependencies not installed")

    # Test get operation
    mock_redis.get.return_value = '{"key": "test-key", "data": "test-value", "created_at": 1625097600.0, "expires_at": null, "metadata": {}}'
    entry = await backend.get("test-key")
    mock_redis.get.assert_called_with("fastadk:test-key")
    assert entry is not None
    assert entry.key == "test-key"
    assert entry.data == "test-value"

    # Test get with missing key
    mock_redis.get.return_value = None
    entry = await backend.get("missing-key")
    assert entry is None

    # Test set operation
    entry = await backend.set("test-key", "test-value")
    assert entry.key == "test-key"
    assert entry.data == "test-value"
    mock_redis.set.assert_called_once()

    # Test set with TTL
    mock_redis.reset_mock()
    entry = await backend.set("test-key", "test-value", ttl_seconds=60)
    assert entry.key == "test-key"
    assert entry.data == "test-value"
    assert entry.expires_at is not None
    mock_redis.setex.assert_called_once()

    # Test delete operation
    mock_redis.reset_mock()
    mock_redis.delete.return_value = 1
    result = await backend.delete("test-key")
    mock_redis.delete.assert_called_with("fastadk:test-key")
    assert result is True

    # Test delete non-existent key
    mock_redis.reset_mock()
    mock_redis.delete.return_value = 0
    result = await backend.delete("missing-key")
    assert result is False

    # Test exists operation
    mock_redis.reset_mock()
    mock_redis.exists.return_value = 1
    mock_redis.get.return_value = '{"key": "test-key", "data": "test-value", "created_at": 1625097600.0, "expires_at": null, "metadata": {}}'
    result = await backend.exists("test-key")
    mock_redis.exists.assert_called_with("fastadk:test-key")
    assert result is True

    # Test keys operation
    mock_redis.reset_mock()
    mock_redis.keys.return_value = ["fastadk:test-key-1", "fastadk:test-key-2"]
    keys = await backend.keys("test-*")
    mock_redis.keys.assert_called_with("fastadk:test-*")
    assert keys == ["test-key-1", "test-key-2"]

    # Test clear operation
    mock_redis.reset_mock()
    mock_redis.keys.return_value = ["fastadk:test-key-1", "fastadk:test-key-2"]
    mock_redis.delete.return_value = 2
    count = await backend.clear("test-*")
    assert count == 2

    # Test health check
    mock_redis.reset_mock()
    mock_redis.ping.return_value = True
    result = await backend.health_check()
    mock_redis.ping.assert_called_once()
    assert result is True
