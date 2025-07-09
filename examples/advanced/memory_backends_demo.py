"""
Memory Backends Demo for FastADK.

This example demonstrates how to use different memory backends in FastADK:
1. InMemory - Simple in-process memory (default)
2. Redis - Persistent memory using Redis
3. Vector - Vector database for semantic search

Usage:
    1. Make sure Redis is running if you want to test the Redis backend:
        - Install: brew install redis
        - Start: redis-server

    2. Run the example:
        uv run examples/advanced/memory_backends_demo.py
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool
from fastadk.memory.base import MemoryBackend

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Message:
    """A simple message class for demonstration purposes."""

    def __init__(self, content: str, sender: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.sender = sender
        self.timestamp = timestamp or datetime.now()

    def __repr__(self) -> str:
        return f"Message({self.sender}: {self.content[:20]}... @ {self.timestamp})"

    def to_dict(self) -> dict:
        """Convert message to dictionary for storage."""
        return {
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        timestamp = (
            datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"]
        )
        return cls(
            content=data["content"],
            sender=data["sender"],
            timestamp=timestamp,
        )


@Agent(
    model="gemini-1.5-pro",
    description="An agent that demonstrates memory backends",
    provider="gemini",  # Will fall back to simulated if no API key is available
    system_prompt="""
    You are a helpful assistant that can store and retrieve information
    using different memory backends.
    """,
)
class MemoryDemoAgent(BaseAgent):
    """Agent that demonstrates different memory backends."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize different memory backends
        from fastadk.memory import InMemoryBackend

        self.in_memory = InMemoryBackend()

        # Initialize Redis memory if Redis is available
        self.redis_memory = None
        self.redis_available = False
        try:
            # Import conditionally to avoid errors if not installed
            try:
                from fastadk.memory import RedisBackend

                # Only attempt to create if import succeeds
                self.redis_memory = RedisBackend(
                    prefix="memory_demo",  # Namespace for this demo
                    default_ttl=3600,  # TTL in seconds
                )
                self.redis_available = True
                logger.info("Redis memory backend initialized")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning("Redis not available: %s", e)
        except Exception as e:
            logger.warning("Redis connection failed: %s", e)

        # Initialize vector memory
        self.vector_memory = None
        self.vector_available = False
        try:
            # Import conditionally to avoid errors if not installed
            try:
                from fastadk.memory import VectorMemoryBackend

                # Only attempt to create if import succeeds
                self.vector_memory = VectorMemoryBackend(
                    vector_store=None,  # Will use default in-memory store
                    embedding_provider=None,  # Will use mock embedding provider
                )
                self.vector_available = True
                logger.info("Vector memory backend initialized")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning("Vector memory not available: %s", e)
        except Exception as e:
            logger.warning("Vector memory initialization failed: %s", e)

        # Track which backend is active
        self.active_memory = "in_memory"

    @tool
    async def switch_memory_backend(self, backend: str) -> dict:
        """
        Switch between memory backends.

        Args:
            backend: The backend to use ('in_memory', 'redis', or 'vector')

        Returns:
            Status of the operation
        """
        backend = backend.lower()
        if backend not in ["in_memory", "redis", "vector"]:
            return {
                "success": False,
                "message": f"Unknown backend: {backend}. Available options: in_memory, redis, vector",
            }

        if backend == "redis" and not self.redis_available:
            return {
                "success": False,
                "message": "Redis backend is not available. Make sure Redis is running.",
            }

        if backend == "vector" and not self.vector_available:
            return {
                "success": False,
                "message": "Vector backend is not available. Make sure vector dependencies are installed.",
            }

        self.active_memory = backend
        return {
            "success": True,
            "message": f"Switched to {backend} backend",
            "features": self._get_backend_features(backend),
        }

    def _get_backend_features(self, backend: str) -> List[str]:
        """Get features of the specified backend."""
        common_features = ["store", "retrieve", "delete"]

        if backend == "in_memory":
            return common_features + ["fast", "non-persistent"]
        elif backend == "redis":
            return common_features + ["persistent", "TTL support", "distributed"]
        elif backend == "vector":
            return common_features + ["semantic search", "embeddings", "similarity"]

        return common_features

    def _get_active_backend(self) -> MemoryBackend:
        """Get the currently active memory backend."""
        if self.active_memory == "in_memory":
            return self.in_memory
        elif self.active_memory == "redis" and self.redis_available:
            return self.redis_memory
        elif self.active_memory == "vector" and self.vector_available:
            return self.vector_memory
        else:
            return self.in_memory

    @tool
    async def store_message(self, content: str, sender: str) -> dict:
        """
        Store a message in the active memory backend.

        Args:
            content: The message content
            sender: The sender of the message

        Returns:
            Status of the operation
        """
        message = Message(content=content, sender=sender)
        message_id = f"msg_{int(time.time())}_{random.randint(1000, 9999)}"

        try:
            backend = self._get_active_backend()

            # For vector backend, we'll add the content as metadata for semantic search
            metadata = {}
            if self.active_memory == "vector":
                metadata["text"] = content

            # Store the message
            await backend.set(
                key=message_id,
                data=message.to_dict(),
                ttl_seconds=3600 if self.active_memory == "redis" else None,
            )

            return {
                "success": True,
                "message": f"Message stored with ID: {message_id}",
                "id": message_id,
                "backend": self.active_memory,
                "timestamp": datetime.now().isoformat(),
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error storing message: %s", e)
            return {
                "success": False,
                "message": f"Error storing message: {str(e)}",
                "backend": self.active_memory,
            }

    @tool
    async def retrieve_message(self, message_id: str) -> dict:
        """
        Retrieve a message from the active memory backend.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            The retrieved message or error status
        """
        try:
            backend = self._get_active_backend()

            # Get the message from memory
            entry = await backend.get(message_id)

            if not entry or entry.data is None:
                return {
                    "success": False,
                    "message": f"Message with ID {message_id} not found",
                    "backend": self.active_memory,
                }

            # Parse the message data
            message = Message.from_dict(entry.data)

            return {
                "success": True,
                "message": "Message retrieved successfully",
                "id": message_id,
                "content": message.content,
                "sender": message.sender,
                "timestamp": message.timestamp.isoformat(),
                "backend": self.active_memory,
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error retrieving message: %s", e)
            return {
                "success": False,
                "message": f"Error retrieving message: {str(e)}",
                "backend": self.active_memory,
            }

    @tool
    async def search_messages(self, query: str, limit: int = 3) -> dict:
        """
        Search for messages in the active memory backend.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            The search results
        """
        try:
            backend = self._get_active_backend()
            results = []

            if self.active_memory == "vector" and self.vector_available:
                # Semantic search for vector memory
                entries = await backend.search_semantic(query, limit=limit)
                for entry in entries:
                    if not entry.data:
                        continue
                    message = Message.from_dict(entry.data)
                    results.append(
                        {
                            "id": entry.key,
                            "content": message.content,
                            "sender": message.sender,
                            "timestamp": message.timestamp.isoformat(),
                            "score": 0.8,  # Example score
                        }
                    )
            else:
                # Simple string matching for other backends
                all_keys = await backend.keys()
                for key in all_keys:
                    entry = await backend.get(key)
                    if not entry or not entry.data:
                        continue

                    message = Message.from_dict(entry.data)
                    if query.lower() in message.content.lower():
                        results.append(
                            {
                                "id": key,
                                "content": message.content,
                                "sender": message.sender,
                                "timestamp": message.timestamp.isoformat(),
                                "score": 1.0,  # No actual scoring for simple search
                            }
                        )
                        if len(results) >= limit:
                            break

            # Sort and limit results
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

            return {
                "success": True,
                "message": f"Found {len(results)} matching messages",
                "results": results,
                "backend": self.active_memory,
                "query": query,
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error searching messages: %s", e)
            return {
                "success": False,
                "message": f"Error searching messages: {str(e)}",
                "backend": self.active_memory,
            }

    @tool
    async def delete_message(self, message_id: str) -> dict:
        """
        Delete a message from the active memory backend.

        Args:
            message_id: The ID of the message to delete

        Returns:
            Status of the operation
        """
        try:
            backend = self._get_active_backend()

            # Delete the message
            success = await backend.delete(message_id)

            if not success:
                return {
                    "success": False,
                    "message": f"Message with ID {message_id} not found or could not be deleted",
                    "backend": self.active_memory,
                }

            return {
                "success": True,
                "message": f"Message with ID {message_id} deleted successfully",
                "backend": self.active_memory,
                "timestamp": datetime.now().isoformat(),
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error deleting message: %s", e)
            return {
                "success": False,
                "message": f"Error deleting message: {str(e)}",
                "backend": self.active_memory,
            }


async def run_memory_demo() -> None:
    """Run the memory backends demonstration."""
    print("\n" + "=" * 60)
    print("💾 FastADK Memory Backends Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    # Initialize the agent
    agent = MemoryDemoAgent()

    # Demo data
    sample_messages = [
        {"content": "The meeting is scheduled for tomorrow at 2pm", "sender": "Alice"},
        {"content": "Please review the proposal before the meeting", "sender": "Bob"},
        {"content": "I'll prepare the presentation slides tonight", "sender": "Alice"},
        {"content": "Don't forget to book the conference room", "sender": "Charlie"},
        {"content": "The client wants to discuss pricing options", "sender": "Bob"},
        {
            "content": "We should highlight our new features in the meeting",
            "sender": "Alice",
        },
        {"content": "I've updated the financial projections", "sender": "David"},
        {
            "content": "The marketing team will join us for the second part",
            "sender": "Charlie",
        },
    ]

    # List of memory backends to demo
    backends = ["in_memory", "redis", "vector"]
    message_ids = {}  # To store message IDs for each backend

    for backend in backends:
        print(f"\n\n📂 Testing {backend.upper()} Backend")
        print("-" * 40)

        # Switch to this backend
        result = await agent.switch_memory_backend(backend=backend)

        # Skip if the backend is not available
        if not result.get("success", False):
            print(f"  ❌ {result.get('message', 'Unknown error')}")
            continue

        print(f"  ✅ Switched to {backend} backend")
        print(f"  📋 Features: {', '.join(result.get('features', []))}")

        # Store messages
        print("\n  📝 Storing messages...")
        backend_message_ids = []

        for i, msg in enumerate(sample_messages, 1):
            result = await agent.store_message(
                content=msg["content"], sender=msg["sender"]
            )

            if result.get("success", False):
                msg_id = result.get("id")
                backend_message_ids.append(msg_id)
                print(f"    ✓ Message {i} stored: {msg_id}")
            else:
                print(f"    ✗ Failed to store message {i}: {result.get('message')}")

        message_ids[backend] = backend_message_ids

        # Retrieve messages
        if backend_message_ids:
            print("\n  🔍 Retrieving messages...")
            for i, msg_id in enumerate(backend_message_ids[:3], 1):
                result = await agent.retrieve_message(message_id=msg_id)

                if result.get("success", False):
                    print(
                        f"    ✓ Message {i} retrieved: {result.get('content')[:30]}..."
                    )
                else:
                    print(
                        f"    ✗ Failed to retrieve message {i}: {result.get('message')}"
                    )

        # Search messages
        print("\n  🔎 Searching messages...")
        search_queries = ["meeting", "presentation", "financial"]

        for query in search_queries:
            result = await agent.search_messages(query=query, limit=2)

            if result.get("success", False):
                print(
                    f"    📌 Search for '{query}': {len(result.get('results', []))} results"
                )
                for i, res in enumerate(result.get("results", []), 1):
                    print(
                        f"      {i}. {res.get('content')[:40]}... (Score: {res.get('score', 0):.2f})"
                    )
            else:
                print(f"    ❌ Search for '{query}' failed: {result.get('message')}")

        # Delete a message
        if backend_message_ids:
            print("\n  🗑️ Deleting a message...")
            msg_id = backend_message_ids[0]
            result = await agent.delete_message(message_id=msg_id)

            if result.get("success", False):
                print(f"    ✓ Message deleted: {msg_id}")

                # Verify deletion
                verify = await agent.retrieve_message(message_id=msg_id)
                if not verify.get("success", False):
                    print(f"    ✓ Verified message is no longer retrievable")
                else:
                    print(f"    ❌ Message still retrievable after deletion")
            else:
                print(f"    ❌ Failed to delete message: {result.get('message')}")

    # Demonstrate vector search capabilities
    if "vector" in message_ids and message_ids["vector"] and agent.vector_available:
        print("\n\n🔍 Demonstrating Vector Search Capabilities")
        print("-" * 60)

        # Switch to vector backend
        await agent.switch_memory_backend(backend="vector")

        # Search with semantic queries
        semantic_queries = [
            "project finances",
            "scheduling a meeting",
            "presentation preparation",
        ]

        print("  🧠 Semantic Search Examples:")
        for query in semantic_queries:
            result = await agent.search_messages(query=query, limit=2)

            if result.get("success", False):
                print(f"\n    📌 Query: '{query}'")
                for i, res in enumerate(result.get("results", []), 1):
                    print(
                        f"      {i}. {res.get('content')} (Score: {res.get('score', 0):.2f})"
                    )
            else:
                print(f"    ❌ Search for '{query}' failed: {result.get('message')}")

    print("\n" + "=" * 60)
    print("🏁 FastADK - Memory Backends Demo Completed")
    print("=" * 60 + "\n")


async def main() -> None:
    """Run the main demo."""
    await run_memory_demo()


if __name__ == "__main__":
    asyncio.run(main())
