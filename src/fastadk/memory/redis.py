"""
Redis memory backend for FastADK.

This module provides a Redis-based memory backend for production use.
It supports all standard memory operations plus semantic search capabilities.
"""

import json
import time
from typing import Any

# Ignore import errors for these modules
try:
    import redis.asyncio as redis
except ImportError:
    # Create placeholder for type checking
    redis = None

from loguru import logger

from fastadk.core.exceptions import OperationError, ServiceConnectionError

from .base import MemoryBackend, MemoryEntry


class RedisBackend(MemoryBackend):
    """
    Redis implementation of the memory backend.

    This backend stores data in Redis, making it suitable for production use
    with persistence, high availability, and scaling capabilities.

    Attributes:
        prefix: Prefix for all keys stored in Redis
        redis: Redis client connection
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "fastadk:",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Redis backend.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional Redis password
            prefix: Key prefix for all FastADK entries
            **kwargs: Additional arguments passed to Redis client
        """
        self.prefix = prefix
        connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "decode_responses": True,  # Always decode to str
        }

        if password:
            connection_params["password"] = password

        # Add any additional params
        connection_params.update(kwargs)

        try:
            if redis is None:
                raise ImportError("Redis module not installed")
            self.redis = redis.Redis(**connection_params)
            logger.debug("Initialized Redis memory backend: %s:%s/%s", host, port, db)
        except Exception as e:
            logger.error("Failed to initialize Redis memory backend: %s", str(e))
            raise ServiceConnectionError(
                message=f"Failed to connect to Redis: {str(e)}",
                error_code="REDIS_CONNECTION_ERROR",
                details={"host": host, "port": port, "db": db},
            ) from e

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Remove prefix from key."""
        if key.startswith(self.prefix):
            return key[len(self.prefix) :]
        return key

    async def get(self, key: str) -> MemoryEntry | None:
        """
        Get a memory entry by key.

        Args:
            key: The key to look up

        Returns:
            The memory entry if found and not expired, otherwise None
        """
        prefixed_key = self._prefixed_key(key)
        try:
            data = await self.redis.get(prefixed_key)
            if data is None:
                return None

            # Parse the JSON data
            entry_dict = json.loads(data)
            entry = MemoryEntry(**entry_dict)

            # Check if the entry is expired
            if entry.is_expired():
                await self.delete(key)
                return None

            return entry
        except Exception as e:
            logger.error("Redis get error for key %s: %s", key, str(e))
            raise OperationError(
                message=f"Failed to get data from Redis: {str(e)}",
                error_code="REDIS_GET_ERROR",
                details={"key": key},
            ) from e

    async def set(
        self, key: str, data: Any, ttl_seconds: int | None = None
    ) -> MemoryEntry:
        """
        Store a value in Redis.

        Args:
            key: The key to store the data under
            data: The data to store
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            The created memory entry
        """
        prefixed_key = self._prefixed_key(key)
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        try:
            # Create memory entry
            entry = MemoryEntry(
                key=key,
                data=data,
                expires_at=expires_at,
                created_at=time.time(),
            )

            # Convert to JSON and store in Redis
            entry_json = entry.model_dump_json()

            # Set with TTL if provided
            if ttl_seconds is not None:
                await self.redis.setex(prefixed_key, ttl_seconds, entry_json)
            else:
                await self.redis.set(prefixed_key, entry_json)

            return entry

        except Exception as e:
            logger.error("Redis set error for key %s: %s", key, str(e))
            raise OperationError(
                message=f"Failed to set data in Redis: {str(e)}",
                error_code="REDIS_SET_ERROR",
                details={"key": key},
            ) from e

    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        prefixed_key = self._prefixed_key(key)
        try:
            result = await self.redis.delete(prefixed_key)
            # Explicitly cast the result to ensure it's a boolean
            return bool(result > 0)
        except Exception as e:
            logger.error("Redis delete error for key %s: %s", key, str(e))
            raise OperationError(
                message=f"Failed to delete data from Redis: {str(e)}",
                error_code="REDIS_DELETE_ERROR",
                details={"key": key},
            ) from e

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        prefixed_key = self._prefixed_key(key)
        try:
            exists = await self.redis.exists(prefixed_key)
            if exists:
                # Make sure it's not expired
                entry = await self.get(key)
                # Explicitly cast the result to ensure it's a boolean
                return entry is not None
            return False
        except Exception as e:
            logger.error("Redis exists error for key %s: %s", key, str(e))
            raise OperationError(
                message=f"Failed to check key existence in Redis: {str(e)}",
                error_code="REDIS_EXISTS_ERROR",
                details={"key": key},
            ) from e

    async def keys(self, pattern: str | None = None) -> list[str]:
        """
        Get all keys matching a pattern.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            List of matching keys (without prefix)
        """
        try:
            if pattern is None:
                redis_pattern = f"{self.prefix}*"
            else:
                redis_pattern = f"{self.prefix}{pattern}"

            keys = await self.redis.keys(redis_pattern)
            # Remove the prefix from each key
            return [self._strip_prefix(key) for key in keys]
        except Exception as e:
            logger.error("Redis keys error for pattern %s: %s", pattern, str(e))
            raise OperationError(
                message=f"Failed to get keys from Redis: {str(e)}",
                error_code="REDIS_KEYS_ERROR",
                details={"pattern": pattern},
            ) from e

    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear entries from Redis.

        Args:
            pattern: Optional pattern to match keys against

        Returns:
            Number of entries cleared
        """
        try:
            keys = await self.keys(pattern)
            if not keys:
                return 0

            # Add prefix back for deletion
            prefixed_keys = [self._prefixed_key(key) for key in keys]
            count = await self.redis.delete(*prefixed_keys)
            # Explicitly cast the result to ensure it's an integer
            return int(count)
        except Exception as e:
            logger.error("Redis clear error for pattern %s: %s", pattern, str(e))
            raise OperationError(
                message=f"Failed to clear keys in Redis: {str(e)}",
                error_code="REDIS_CLEAR_ERROR",
                details={"pattern": pattern},
            ) from e

    async def ttl(self, key: str) -> float | None:
        """
        Get the time-to-live for a key in seconds.

        Args:
            key: The key to check

        Returns:
            The TTL in seconds, or None if the key doesn't exist or has no expiration
        """
        try:
            # Check if entry exists first
            entry = await self.get(key)
            if entry is None:
                return None

            # Use entry's TTL method
            return entry.get_ttl()
        except Exception as e:
            logger.error("Redis TTL error for key %s: %s", key, str(e))
            raise OperationError(
                message=f"Failed to get TTL in Redis: {str(e)}",
                error_code="REDIS_TTL_ERROR",
                details={"key": key},
            ) from e

    async def search_semantic(
        self, query: str, limit: int = 10, threshold: float = 0.0
    ) -> list[MemoryEntry]:
        """
        Basic search implementation for Redis.

        Note: This is a simple implementation that just checks for string containment.
        For true semantic search, consider using Redis Stack with RediSearch and vector
        capabilities, or a dedicated vector database.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of matching memory entries
        """
        try:
            # Get all keys
            all_keys = await self.keys()
            results = []
            query_lower = query.lower()

            # For each key, check if the data contains the query
            for key in all_keys:
                entry = await self.get(key)
                if entry is None:
                    continue

                # Calculate a simple similarity score (0-1) based on substring match
                similarity = 0.0

                # Check if string data contains query
                if isinstance(entry.data, str) and query_lower in entry.data.lower():
                    # Simple similarity score based on relative length of query vs content
                    similarity = len(query_lower) / max(len(entry.data.lower()), 1)
                # For dictionaries, check values
                elif isinstance(entry.data, dict):
                    for value in entry.data.values():
                        if isinstance(value, str) and query_lower in value.lower():
                            similarity = len(query_lower) / max(len(value.lower()), 1)
                            break

                # Only include if above threshold
                if similarity >= threshold:
                    results.append(entry)
                    if len(results) >= limit:
                        break

            return results[:limit]
        except Exception as e:
            logger.error("Redis search error for query %s: %s", query, str(e))
            raise OperationError(
                message=f"Failed to search in Redis: {str(e)}",
                error_code="REDIS_SEARCH_ERROR",
                details={"query": query},
            ) from e

    async def health_check(self) -> bool:
        """
        Check if the Redis backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple ping to check connection
            result = await self.redis.ping()
            # Explicitly cast the result to ensure it's a boolean
            return bool(result)
        except redis.RedisError:
            logger.error("Redis health check failed")
            return False
