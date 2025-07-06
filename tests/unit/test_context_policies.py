"""
Tests for context policies.

This module tests the different context policies that manage conversation history.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from fastadk.core.context_policy import (
    HybridVectorRetrievalPolicy,
    MostRecentPolicy,
    SummarizeOlderPolicy,
)


@dataclass
class MockContextEntry:
    """Mock context entry for testing."""

    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockSummarizer:
    """Mock summarizer for testing."""

    async def summarize(self, entries: List[MockContextEntry]) -> MockContextEntry:
        """Create a mock summary."""
        return MockContextEntry(
            role="system",
            content=f"Summary of {len(entries)} messages",
            metadata={"is_summary": True},
        )


class TestMostRecentPolicy:
    """Tests for the MostRecentPolicy class."""

    @pytest.mark.asyncio
    async def test_apply_empty_history(self):
        """Test applying policy to empty history."""
        policy = MostRecentPolicy(max_messages=5)
        result = await policy.apply([])
        assert result == []

    @pytest.mark.asyncio
    async def test_apply_small_history(self):
        """Test applying policy to history smaller than max messages."""
        policy = MostRecentPolicy(max_messages=5)

        # Create a small history
        history = [
            MockContextEntry(role="user", content=f"Message {i}") for i in range(3)
        ]

        result = await policy.apply(history)
        assert result == history  # Should return all messages

    @pytest.mark.asyncio
    async def test_apply_large_history(self):
        """Test applying policy to history larger than max messages."""
        policy = MostRecentPolicy(max_messages=5)

        # Create a large history
        history = [
            MockContextEntry(role="user", content=f"Message {i}") for i in range(10)
        ]

        result = await policy.apply(history)
        assert len(result) == 5
        assert result == history[-5:]  # Should return only the 5 most recent


class TestSummarizeOlderPolicy:
    """Tests for the SummarizeOlderPolicy class."""

    @pytest.mark.asyncio
    async def test_apply_empty_history(self):
        """Test applying policy to empty history."""
        policy = SummarizeOlderPolicy(
            threshold_tokens=1000,
            max_recent_messages=5,
            summarizer=MockSummarizer(),
        )
        result = await policy.apply([])
        assert result == []

    @pytest.mark.asyncio
    async def test_apply_under_threshold(self):
        """Test applying policy to history under the token threshold."""
        policy = SummarizeOlderPolicy(
            threshold_tokens=1000,  # High threshold
            max_recent_messages=5,
            summarizer=MockSummarizer(),
        )

        # Create a small history with short messages
        history = [
            MockContextEntry(role="user", content=f"Short message {i}")
            for i in range(10)
        ]

        result = await policy.apply(history)
        assert result == history  # Should return all messages without summarizing

    @pytest.mark.asyncio
    async def test_apply_over_threshold(self):
        """Test applying policy to history over the token threshold."""
        mock_summarizer = MockSummarizer()
        policy = SummarizeOlderPolicy(
            threshold_tokens=10,  # Low threshold
            max_recent_messages=3,
            summarizer=mock_summarizer,
        )

        # Create a history with long messages
        history = [
            MockContextEntry(
                role="user",
                content=f"This is a longer message with more tokens {i} " * 5,
            )
            for i in range(5)
        ]

        result = await policy.apply(history)
        assert len(result) == 4  # Summary + 3 recent messages
        assert result[0].role == "system"  # First message should be the summary
        assert "Summary of 2 messages" in result[0].content
        assert (
            result[1:] == history[-3:]
        )  # Remaining messages should be the 3 most recent

    @pytest.mark.asyncio
    async def test_apply_without_summarizer(self):
        """Test applying policy without a summarizer."""
        policy = SummarizeOlderPolicy(
            threshold_tokens=10,  # Low threshold
            max_recent_messages=3,
            summarizer=None,  # No summarizer
        )

        # Create a history with long messages
        history = [
            MockContextEntry(
                role="user",
                content=f"This is a longer message with more tokens {i} " * 5,
            )
            for i in range(5)
        ]

        result = await policy.apply(history)
        assert len(result) == 3  # Only the 3 most recent messages
        assert result == history[-3:]  # Should return only the 3 most recent


class TestHybridVectorRetrievalPolicy:
    """Tests for the HybridVectorRetrievalPolicy class."""

    @pytest.mark.asyncio
    async def test_apply_empty_history(self):
        """Test applying policy to empty history."""
        policy = HybridVectorRetrievalPolicy(
            vector_k=2,
            max_recent_messages=3,
        )
        result = await policy.apply([])
        assert result == []

    @pytest.mark.asyncio
    async def test_apply_small_history(self):
        """Test applying policy to history smaller than max messages."""
        policy = HybridVectorRetrievalPolicy(
            vector_k=2,
            max_recent_messages=5,
        )

        # Create a small history
        history = [
            MockContextEntry(role="user", content=f"Message {i}") for i in range(3)
        ]

        result = await policy.apply(history)
        assert result == history  # Should return all messages

    @pytest.mark.asyncio
    async def test_apply_large_history_without_user_message(self):
        """Test applying policy to history without recent user message."""
        policy = HybridVectorRetrievalPolicy(
            vector_k=2,
            max_recent_messages=3,
        )

        # Create a large history with no user messages in the recent set
        history = [
            MockContextEntry(role="user", content=f"Message {i}") for i in range(5)
        ] + [
            MockContextEntry(role="assistant", content=f"Response {i}")
            for i in range(5, 10)
        ]

        result = await policy.apply(history)
        assert len(result) == 3  # Should return only the 3 most recent
        assert result == history[-3:]

    @pytest.mark.asyncio
    async def test_apply_large_history_with_user_message(self):
        """Test applying policy to history with recent user message."""
        policy = HybridVectorRetrievalPolicy(
            vector_k=2,
            max_recent_messages=3,
            similarity_threshold=0.0,  # Accept any similarity
        )

        # Create history with relevant older messages
        history = [
            MockContextEntry(role="user", content="Tell me about Python programming"),
            MockContextEntry(
                role="assistant", content="Python is a popular programming language"
            ),
            MockContextEntry(role="user", content="What are Python libraries?"),
            MockContextEntry(
                role="assistant", content="Python libraries are reusable code packages"
            ),
            MockContextEntry(role="user", content="Tell me about weather"),
            MockContextEntry(
                role="assistant", content="Weather is the state of the atmosphere"
            ),
            MockContextEntry(
                role="user", content="Tell me more about Python libraries"
            ),
        ]

        result = await policy.apply(history)
        assert len(result) >= 3  # Should include recent messages plus relevant ones
        # The last message should always be included
        assert result[-1] == history[-1]
        # We should have some relevant messages about Python libraries
        assert any("Python libraries" in entry.content for entry in result)

    @pytest.mark.asyncio
    async def test_apply_with_metadata_filter(self):
        """Test applying policy with metadata filter."""
        policy = HybridVectorRetrievalPolicy(
            vector_k=2,
            max_recent_messages=3,
            similarity_threshold=0.0,  # Accept any similarity
            metadata_filter={"important": True},
        )

        # Create history with some important messages
        history = [
            MockContextEntry(
                role="user", content="Important info", metadata={"important": True}
            ),
            MockContextEntry(role="assistant", content="Noted the important info"),
            MockContextEntry(role="user", content="Regular message"),
            MockContextEntry(role="assistant", content="Regular response"),
            MockContextEntry(
                role="user",
                content="Another important thing",
                metadata={"important": True},
            ),
            MockContextEntry(
                role="assistant", content="Noted the other important thing"
            ),
            MockContextEntry(
                role="user", content="Tell me about all the important things"
            ),
        ]

        result = await policy.apply(history)
        # Should include recent messages plus ones marked as important
        assert len(result) >= 3
        # Check that filtered messages are included
        important_entries = [
            entry for entry in result if entry.metadata.get("important") is True
        ]
        assert len(important_entries) > 0
