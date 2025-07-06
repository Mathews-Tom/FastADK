"""
Caching system for FastADK.

This module provides a flexible caching mechanism for reusing responses from LLMs
and tool calls. It includes an in-memory LRU cache and support for Redis-based
distributed caching.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, cast

# Optional Redis support
try:
    import redis
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("fastadk.cache")

# Type variables for cache key and value
K = TypeVar("K")
V = TypeVar("V")


class CacheBackend(Generic[K, V], ABC):
    """
    Abstract base class for cache backends.

    Cache backends handle the storage and retrieval of cached values.
    """

    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value, or None if the key is not in the cache
        """

    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds
        """

    @abstractmethod
    async def delete(self, key: K) -> None:
        """
        Delete a value from the cache.

        Args:
            key: The cache key to delete
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from the cache."""

    @abstractmethod
    async def contains(self, key: K) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists, False otherwise
        """


class InMemoryCache(CacheBackend[K, V]):
    """
    In-memory LRU cache implementation.

    This cache stores values in memory with an optional TTL, and automatically
    evicts the least recently used items when the cache reaches its maximum size.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the in-memory cache.

        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[K, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value, or None if the key is not in the cache or has expired
        """
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            current_time = time.time()

            # Check if the entry has expired
            if "expires_at" in entry and entry["expires_at"] <= current_time:
                # Remove expired entry
                del self._cache[key]
                return None

            # Move the entry to the end (most recently used)
            self._cache.move_to_end(key)
            return cast(V, entry["value"])

    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds
        """
        async with self._lock:
            # Calculate expiration time if TTL is provided
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl

            # Create or update the cache entry
            entry = {"value": value}
            if expires_at is not None:
                entry["expires_at"] = expires_at

            # Add or update the entry
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                # Move to the end (most recently used)
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                # If we've exceeded the max size, remove the oldest entry
                if len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)  # Remove the first item (oldest)

    async def delete(self, key: K) -> None:
        """
        Delete a value from the cache.

        Args:
            key: The cache key to delete
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear all values from the cache."""
        async with self._lock:
            self._cache.clear()

    async def contains(self, key: K) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists and has not expired, False otherwise
        """
        async with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            current_time = time.time()

            # Check if the entry has expired
            if "expires_at" in entry and entry["expires_at"] <= current_time:
                # Remove expired entry
                del self._cache[key]
                return False

            return True


class RedisCache(CacheBackend[str, str]):
    """
    Redis-based distributed cache implementation.

    This cache uses Redis for storage, allowing for distributed caching
    across multiple instances of FastADK.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "fastadk:cache:",
    ):
        """
        Initialize the Redis cache.

        Args:
            redis_url: Redis connection URL
            prefix: Prefix for cache keys in Redis
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support requires the redis package. "
                "Install it with: uv add redis[hiredis]"
            )

        self.prefix = prefix
        self.redis: Optional[Redis] = None
        self.redis_url = redis_url

    async def initialize(self) -> None:
        """Initialize the Redis connection."""
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support requires the redis package. "
                "Install it with: uv add redis[hiredis]"
            )

        self.redis = cast(Redis, Redis.from_url(self.redis_url))
        try:
            # Test the connection
            await self.redis.ping()
            logger.info("Successfully connected to Redis at %s", self.redis_url)
        except redis.RedisError as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.redis = None
            raise

    def _get_full_key(self, key: str) -> str:
        """
        Get the full Redis key with prefix.

        Args:
            key: The cache key

        Returns:
            The full key with prefix
        """
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value from the Redis cache.

        Args:
            key: The cache key

        Returns:
            The cached value, or None if the key is not in the cache
        """
        if not self.redis:
            logger.warning("Redis cache not initialized")
            return None

        try:
            full_key = self._get_full_key(key)
            value = await self.redis.get(full_key)
            return value.decode("utf-8") if value else None
        except redis.RedisError as e:
            logger.error("Redis error on get: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """
        Set a value in the Redis cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds
        """
        if not self.redis:
            logger.warning("Redis cache not initialized")
            return

        try:
            full_key = self._get_full_key(key)
            if ttl is not None:
                await self.redis.setex(full_key, ttl, value)
            else:
                await self.redis.set(full_key, value)
        except redis.RedisError as e:
            logger.error("Redis error on set: %s", e)

    async def delete(self, key: str) -> None:
        """
        Delete a value from the Redis cache.

        Args:
            key: The cache key to delete
        """
        if not self.redis:
            logger.warning("Redis cache not initialized")
            return

        try:
            full_key = self._get_full_key(key)
            await self.redis.delete(full_key)
        except redis.RedisError as e:
            logger.error("Redis error on delete: %s", e)

    async def clear(self) -> None:
        """Clear all values with this cache's prefix from Redis."""
        if not self.redis:
            logger.warning("Redis cache not initialized")
            return

        try:
            # Find all keys with our prefix
            cursor = b"0"
            pattern = f"{self.prefix}*"
            count = 0

            while cursor:
                cursor, keys = await self.redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    await self.redis.delete(*keys)
                    count += len(keys)

                # Convert cursor back to string if needed
                cursor = cursor if isinstance(cursor, bytes) else cursor.encode("utf-8")
                if cursor == b"0":
                    break

            logger.info("Cleared %d cached items with prefix %s", count, self.prefix)
        except redis.RedisError as e:
            logger.error("Redis error on clear: %s", e)

    async def contains(self, key: str) -> bool:
        """
        Check if a key exists in the Redis cache.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists, False otherwise
        """
        if not self.redis:
            logger.warning("Redis cache not initialized")
            return False

        try:
            full_key = self._get_full_key(key)
            exists_count = await self.redis.exists(full_key)
            return bool(exists_count > 0)
        except redis.RedisError as e:
            logger.error("Redis error on contains: %s", e)
            return False


class CacheManager:
    """
    Centralized cache manager for FastADK.

    This class provides a unified interface for caching operations,
    with support for different backend storage options.
    """

    def __init__(
        self,
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379/0",
        max_memory_size: int = 1000,
        namespace: str = "fastadk",
    ):
        """
        Initialize the cache manager.

        Args:
            backend: Cache backend to use ('memory' or 'redis')
            redis_url: Redis connection URL (used only if backend is 'redis')
            max_memory_size: Maximum size for in-memory cache
            namespace: Namespace for cache keys
        """
        self.backend_type = backend
        self.namespace = namespace
        self.cache: CacheBackend

        if backend == "memory":
            self.cache = InMemoryCache(max_size=max_memory_size)
        elif backend == "redis":
            self.cache = RedisCache(redis_url=redis_url, prefix=f"{namespace}:")
        else:
            raise ValueError(f"Unsupported cache backend: {backend}")

        self._initialized = False
        logger.info("Initialized cache manager with %s backend", backend)

    async def initialize(self) -> None:
        """Initialize the cache backend if needed."""
        if not self._initialized:
            if isinstance(self.cache, RedisCache):
                await self.cache.initialize()
            self._initialized = True

    def _make_key(self, key_parts: Any) -> str:
        """
        Create a deterministic cache key from the provided parts.

        Args:
            key_parts: Any serializable object to use for key generation

        Returns:
            A string cache key
        """
        # Serialize the key parts to a JSON string
        if isinstance(key_parts, str):
            serialized = key_parts
        else:
            try:
                serialized = json.dumps(key_parts, sort_keys=True)
            except (TypeError, ValueError):
                # If serialization fails, use string representation
                serialized = str(key_parts)

        # Create a hash of the serialized string
        key_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{key_hash}"

    async def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: The cache key (will be serialized)

        Returns:
            The cached value, or None if not found
        """
        await self.initialize()
        cache_key = self._make_key(key)

        # Get from cache
        value = await self.cache.get(cache_key)
        if value is None:
            return None

        # Deserialize the value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If not JSON, return as is
            return value

    async def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key (will be serialized)
            value: The value to cache (will be serialized)
            ttl: Optional time-to-live in seconds
        """
        await self.initialize()
        cache_key = self._make_key(key)

        # Serialize the value
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            # If serialization fails, convert to string
            serialized = str(value)

        # Store in cache
        await self.cache.set(cache_key, serialized, ttl)

    async def delete(self, key: Any) -> None:
        """
        Delete a value from the cache.

        Args:
            key: The cache key (will be serialized)
        """
        await self.initialize()
        cache_key = self._make_key(key)
        await self.cache.delete(cache_key)

    async def clear(self) -> None:
        """Clear all values from the cache."""
        await self.initialize()
        await self.cache.clear()

    async def contains(self, key: Any) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: The cache key (will be serialized)

        Returns:
            True if the key exists, False otherwise
        """
        await self.initialize()
        cache_key = self._make_key(key)
        return await self.cache.contains(cache_key)


# Global cache manager instance with default settings
default_cache_manager = CacheManager()


def cached(
    ttl: Optional[int] = None,
    key_builder: Optional[Callable[..., Any]] = None,
    cache_manager: Optional[CacheManager] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for caching function results.

    This decorator can be applied to any function or coroutine to cache its
    results. It supports both synchronous and asynchronous functions.

    Args:
        ttl: Optional time-to-live in seconds
        key_builder: Optional function to build the cache key
        cache_manager: Optional custom cache manager instance

    Returns:
        A decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Get the signature for building default cache keys
        sig = inspect.signature(func)
        cache_mgr = cache_manager or default_cache_manager

        # Determine if the function is a coroutine
        is_coroutine = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build the cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key building: function name + bound arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                cache_key = {
                    "func": func.__module__ + "." + func.__qualname__,
                    "args": bound_args.arguments,
                }

            # Try to get from cache
            cached_result = await cache_mgr.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for %s", func.__qualname__)
                return cached_result

            logger.debug("Cache miss for %s", func.__qualname__)

            # Call the function
            if is_coroutine:
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            # Cache the result
            await cache_mgr.set(cache_key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For synchronous functions, we use asyncio.run to bridge
            # with our async cache implementation
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create a new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return the appropriate wrapper based on whether the function is a coroutine
        return async_wrapper if is_coroutine else sync_wrapper

    return decorator
