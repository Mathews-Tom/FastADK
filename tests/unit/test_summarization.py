"""
Tests for summarization services.

This module tests the summarization service that creates summaries of conversation histories.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fastadk.core.summarization import (
    LLMSummarizer,
    SummarizationOptions,
)
from fastadk.memory.inmemory import InMemoryBackend


@dataclass
class MockContextEntry:
    """Mock context entry for testing."""

    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    async def generate(self, messages, **kwargs):
        """Generate a mock response."""
        # Return a simple summary based on the number of messages
        message_count = len(messages) - 1  # Subtract the system message
        return (
            f"Summarized {message_count} messages. The conversation was about testing."
        )


class TestLLMSummarizer:
    """Tests for the LLMSummarizer class."""

    @pytest_asyncio.fixture
    async def memory(self):
        """Create a memory backend for testing."""
        return InMemoryBackend()

    @pytest_asyncio.fixture
    async def llm_provider(self):
        """Create a mock LLM provider."""
        return MockLLMProvider()

    @pytest_asyncio.fixture
    async def summarizer(self, llm_provider, memory):
        """Create an LLM summarizer for testing."""
        return LLMSummarizer(
            llm_provider=llm_provider,
            memory_backend=memory,
            cache_ttl=60,
        )

    @pytest.mark.asyncio
    async def test_summarize_empty_entries(self, summarizer):
        """Test summarizing empty entries."""
        result = await summarizer.summarize([])
        assert result.role == "system"
        assert "No previous conversation" in result.content

    @pytest.mark.asyncio
    async def test_summarize_with_entries(self, summarizer):
        """Test summarizing entries."""
        # Create some entries
        entries = [
            MockContextEntry(role="user", content="Hello"),
            MockContextEntry(role="assistant", content="Hi there"),
            MockContextEntry(role="user", content="How are you?"),
            MockContextEntry(role="assistant", content="I'm doing well, thanks!"),
        ]

        result = await summarizer.summarize(entries)
        assert result.role == "system"
        assert "Previous conversation summary" in result.content
        assert "Summarized 4 messages" in result.content
        assert "testing" in result.content
        assert result.metadata.get("is_summary") is True
        assert result.metadata.get("summarized_count") == 4

    @pytest.mark.asyncio
    async def test_summarize_with_options(self, summarizer):
        """Test summarizing with custom options."""
        # Create some entries
        entries = [
            MockContextEntry(role="user", content="Hello"),
            MockContextEntry(role="assistant", content="Hi there"),
        ]

        # Use custom options
        options = SummarizationOptions(
            max_length=50,
            style="bullets",
            include_entities=True,
        )

        result = await summarizer.summarize(entries, options=options)
        assert result.role == "system"
        assert "Previous conversation summary" in result.content

        # The mock provider doesn't use the options, but in a real
        # implementation, these would affect the summary format

    @pytest.mark.asyncio
    async def test_caching(self, summarizer, memory):
        """Test that summaries are cached."""
        # Create entries with unique IDs to ensure cache key stability
        entries = [
            MockContextEntry(
                id="1",
                role="user",
                content="Hello",
                timestamp=1000.0,
            ),
            MockContextEntry(
                id="2",
                role="assistant",
                content="Hi there",
                timestamp=1001.0,
            ),
        ]

        # First call should generate a summary
        result1 = await summarizer.summarize(entries)

        # Mock the LLM to verify it doesn't get called
        summarizer.llm = AsyncMock()
        summarizer.llm.generate.return_value = "This shouldn't be used"

        # Second call with same entries should use cache
        result2 = await summarizer.summarize(entries)

        # Verify the LLM wasn't called
        summarizer.llm.generate.assert_not_called()

        # Results should be equal
        assert result1.content == result2.content

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, summarizer):
        """Test fallback when LLM errors."""
        # Create some entries
        entries = [
            MockContextEntry(role="user", content="Hello"),
            MockContextEntry(role="assistant", content="Hi there"),
        ]

        # Make the LLM fail
        summarizer.llm = AsyncMock()
        summarizer.llm.generate.side_effect = Exception("LLM error")

        # Should fall back to basic summary
        result = await summarizer.summarize(entries)
        assert result.role == "system"
        assert "Previous conversation summary" in result.content
        assert "is_fallback" in result.metadata
        assert result.metadata.get("is_fallback") is True
