"""
Tests for context management functionality.

This module tests the conversation context management capabilities.
"""

import pytest

from fastadk.core.context import (
    ContextEntry,
    ContextManager,
    ContextWindow,
    ConversationContext,
)
from fastadk.core.exceptions import OperationError
from fastadk.memory.inmemory import InMemoryBackend


class TestContextEntry:
    """Tests for the ContextEntry class."""

    def test_create_entry(self):
        """Test creating a context entry."""
        entry = ContextEntry(role="user", content="Hello")
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.id is not None
        assert entry.timestamp is not None
        assert entry.metadata == {}

    def test_create_entry_with_metadata(self):
        """Test creating a context entry with metadata."""
        metadata = {"client": "web", "locale": "en-US"}
        entry = ContextEntry(role="user", content="Hello", metadata=metadata)
        assert entry.metadata == metadata


class TestContextWindow:
    """Tests for the ContextWindow class."""

    def test_add_entry(self):
        """Test adding entries to the context window."""
        window = ContextWindow(max_entries=3)
        entry1 = ContextEntry(role="user", content="Hello")
        entry2 = ContextEntry(role="assistant", content="Hi there!")

        window.add_entry(entry1)
        window.add_entry(entry2)

        assert len(window.entries) == 2
        assert window.entries[0] == entry1
        assert window.entries[1] == entry2

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        window = ContextWindow(max_entries=2)

        # Add three entries
        window.add_entry(ContextEntry(role="user", content="Hello"))
        window.add_entry(ContextEntry(role="assistant", content="Hi there!"))
        window.add_entry(ContextEntry(role="user", content="How are you?"))

        # Window should only contain the last two
        assert len(window.entries) == 2
        assert window.entries[0].content == "Hi there!"
        assert window.entries[1].content == "How are you?"

    def test_get_entries_by_role(self):
        """Test filtering entries by role."""
        window = ContextWindow()
        window.add_entry(ContextEntry(role="user", content="Hello"))
        window.add_entry(ContextEntry(role="assistant", content="Hi there!"))
        window.add_entry(ContextEntry(role="user", content="How are you?"))
        window.add_entry(ContextEntry(role="system", content="System message"))

        user_entries = window.get_entries(roles=["user"])
        assert len(user_entries) == 2
        assert all(entry.role == "user" for entry in user_entries)

        mixed_entries = window.get_entries(roles=["user", "system"])
        assert len(mixed_entries) == 3
        assert all(entry.role in ["user", "system"] for entry in mixed_entries)


class TestConversationContext:
    """Tests for the ConversationContext class."""

    @pytest.fixture
    def memory_backend(self):
        """Create a memory backend for testing."""
        return InMemoryBackend()

    @pytest.fixture
    def context(self, memory_backend):
        """Create a conversation context for testing."""
        return ConversationContext(
            session_id="test-session",
            memory_backend=memory_backend,
            window_size=5,
        )

    @pytest.mark.asyncio
    async def test_add_entry(self, context):
        """Test adding entries to the context."""
        entry = await context.add_entry(role="user", content="Hello")

        assert entry.role == "user"
        assert entry.content == "Hello"

        # Entry should be in the window
        window_entries = context.window.entries
        assert len(window_entries) == 1
        assert window_entries[0].id == entry.id

        # Entry should also be in full history
        assert len(context._full_history) == 1
        assert context._full_history[0].id == entry.id

    @pytest.mark.asyncio
    async def test_get_entries(self, context):
        """Test retrieving entries from the context."""
        await context.add_entry(role="user", content="Hello")
        await context.add_entry(role="assistant", content="Hi there!")
        await context.add_entry(role="user", content="How are you?")

        # Get all entries
        entries = await context.get_entries()
        assert len(entries) == 3

        # Get entries by role
        user_entries = await context.get_entries(roles=["user"])
        assert len(user_entries) == 2
        assert all(entry.role == "user" for entry in user_entries)

    @pytest.mark.asyncio
    async def test_formatted_context(self, context):
        """Test getting formatted context for LLM."""
        await context.add_entry(role="system", content="You are a helpful assistant")
        await context.add_entry(role="user", content="Hello")
        await context.add_entry(role="assistant", content="Hi there!")

        formatted = await context.get_formatted_context()
        assert len(formatted) == 3
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_persistence(self, context, memory_backend):
        """Test that context is persisted to memory backend."""
        # Add some entries
        await context.add_entry(role="user", content="Hello")
        await context.add_entry(role="assistant", content="Hi there!")

        # Create a new context with the same session ID
        new_context = ConversationContext(
            session_id="test-session",
            memory_backend=memory_backend,
        )

        # Load from memory
        await new_context.load()

        # Should have the same entries
        entries = await new_context.get_entries()
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[0].content == "Hello"
        assert entries[1].role == "assistant"
        assert entries[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_clear(self, context):
        """Test clearing the context."""
        # Add some entries
        await context.add_entry(role="user", content="Hello")
        await context.add_entry(role="assistant", content="Hi there!")

        # Clear the context
        await context.clear()

        # Should have no entries
        entries = await context.get_entries()
        assert len(entries) == 0
        assert len(context._full_history) == 0

    @pytest.mark.asyncio
    async def test_search_semantic(self, context):
        """Test semantic search functionality."""
        await context.add_entry(role="user", content="What is machine learning?")
        await context.add_entry(
            role="assistant",
            content="Machine learning is a subset of AI that focuses on data and algorithms.",
        )
        await context.add_entry(
            role="user", content="How does it differ from deep learning?"
        )

        # Search for "machine learning"
        results = await context.search_semantic("machine learning")
        assert len(results) > 0
        assert "machine learning" in results[0].content.lower()

        # Search for something not in the context
        results = await context.search_semantic("quantum computing")
        assert len(results) == 0

    def test_generate_session_id(self):
        """Test generating session IDs."""
        # Without user ID
        session_id1 = ConversationContext.generate_session_id()
        assert session_id1.startswith("session_")

        # With user ID
        session_id2 = ConversationContext.generate_session_id(user_id="user123")
        assert session_id2.startswith("session_")

        # Different user IDs should generate different session IDs
        session_id3 = ConversationContext.generate_session_id(user_id="user456")
        assert session_id2 != session_id3


class TestContextManager:
    """Tests for the ContextManager class."""

    @pytest.fixture
    def memory_backend(self):
        """Create a memory backend for testing."""
        return InMemoryBackend()

    @pytest.fixture
    def manager(self, memory_backend):
        """Create a context manager for testing."""
        return ContextManager(memory_backend=memory_backend)

    @pytest.mark.asyncio
    async def test_get_context_create(self, manager):
        """Test getting a context that doesn't exist."""
        context = await manager.get_context("new-session", create_if_missing=True)
        assert context.get_session_id() == "new-session"
        assert context._is_loaded

    @pytest.mark.asyncio
    async def test_get_context_error(self, manager):
        """Test getting a context that doesn't exist without creating it."""
        with pytest.raises(OperationError):
            await manager.get_context("missing-session", create_if_missing=False)

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager, memory_backend):
        """Test listing session IDs."""
        # Create some contexts
        context1 = await manager.get_context("session1")
        context2 = await manager.get_context("session2")

        # Add an entry to each so they're saved
        await context1.add_entry(role="user", content="Hello from session 1")
        await context2.add_entry(role="user", content="Hello from session 2")

        # List sessions
        sessions = await manager.list_sessions()
        assert "session1" in sessions
        assert "session2" in sessions

    @pytest.mark.asyncio
    async def test_delete_session(self, manager):
        """Test deleting a session."""
        # Create a context
        context = await manager.get_context("temp-session")
        await context.add_entry(role="user", content="Temporary message")

        # Delete it
        result = await manager.delete_session("temp-session")
        assert result is True

        # Try to get it without creating
        with pytest.raises(OperationError):
            await manager.get_context("temp-session", create_if_missing=False)

        # Delete non-existent session
        result = await manager.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """Test creating a new session."""
        # Create a session with a system message
        context = await manager.create_session(
            user_id="user123", system_message="You are a helpful assistant"
        )

        # Should have a session ID
        assert context.get_session_id() is not None

        # Should have the system message
        entries = await context.get_entries()
        assert len(entries) == 1
        assert entries[0].role == "system"
        assert entries[0].content == "You are a helpful assistant"
