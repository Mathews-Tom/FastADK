"""
Context policy for FastADK.

This module provides the abstract base class and implementations for context policies.
Context policies determine how to manage conversation history, including which messages
to include in the prompt sent to the LLM and how to summarize older messages.
"""

import abc
from typing import Any, Dict, List, Optional, Protocol, TypeVar

# Forward reference
T = TypeVar("T")


class ContextPolicy(abc.ABC):
    """
    Abstract base class for context policies.

    A context policy determines how to manage the conversation history
    for an agent, including which messages to include in the context
    window and when/how to summarize older messages.
    """

    @abc.abstractmethod
    async def apply(self, history: List[Any]) -> List[Any]:
        """
        Apply the policy to a conversation history.

        Args:
            history: The full conversation history

        Returns:
            The filtered/processed history to include in the context
        """


class SummarizerProtocol(Protocol):
    """Protocol for summarization services."""

    async def summarize(self, entries: List[Any]) -> Any:
        """
        Summarize a list of context entries.

        Args:
            entries: The entries to summarize

        Returns:
            A new context entry containing the summary
        """


class MostRecentPolicy(ContextPolicy):
    """
    Policy that keeps only the N most recent messages.

    This is the simplest policy - it just keeps the most recent N messages
    and discards older ones completely.
    """

    def __init__(self, max_messages: int = 10):
        """
        Initialize the policy.

        Args:
            max_messages: Maximum number of messages to keep
        """
        self.max_messages = max_messages

    async def apply(self, history: List[Any]) -> List[Any]:
        """
        Apply the policy to a conversation history.

        Args:
            history: The full conversation history

        Returns:
            The most recent N messages
        """
        return history[-self.max_messages :] if history else []


class SummarizeOlderPolicy(ContextPolicy):
    """
    Policy that summarizes older messages when the total token count exceeds a threshold.

    This policy keeps the most recent messages as-is, but summarizes older messages
    when the total token count exceeds a specified threshold.
    """

    def __init__(
        self,
        threshold_tokens: int,
        max_recent_messages: int = 10,
        summarizer: Optional[SummarizerProtocol] = None,
    ):
        """
        Initialize the policy.

        Args:
            threshold_tokens: Token threshold that triggers summarization
            max_recent_messages: Maximum number of recent messages to keep verbatim
            summarizer: Service to use for summarization
        """
        self.threshold_tokens = threshold_tokens
        self.max_recent_messages = max_recent_messages
        self.summarizer = summarizer
        self._token_estimator = self._estimate_tokens

    async def apply(self, history: List[Any]) -> List[Any]:
        """
        Apply the policy to a conversation history.

        Args:
            history: The full conversation history

        Returns:
            A list with summary entry (if needed) followed by recent messages
        """
        if not history:
            return []

        # If we're under the threshold or don't have enough messages, return all
        total_tokens = sum(self._token_estimator(entry) for entry in history)
        if (
            total_tokens <= self.threshold_tokens
            or len(history) <= self.max_recent_messages
        ):
            return history

        # Split into "older" and "recent" messages
        older_messages = history[: -self.max_recent_messages]
        recent_messages = history[-self.max_recent_messages :]

        # If no summarizer is available, just return recent messages
        if not self.summarizer:
            return recent_messages

        # Summarize older messages
        summary_entry = await self.summarizer.summarize(older_messages)
        return [summary_entry] + recent_messages

    def _estimate_tokens(self, entry: Any) -> int:
        """
        Estimate the number of tokens in a context entry.

        This is a simple estimation based on word count. For more accurate
        token counting, this would be replaced with a proper tokenizer.

        Args:
            entry: The context entry to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: assume average of 4 characters per token
        return len(entry.content) // 4 + 1


class HybridVectorRetrievalPolicy(ContextPolicy):
    """
    Policy that combines recent messages with semantically relevant older messages.

    This policy keeps the most recent messages and retrieves semantically
    relevant older messages based on the current context.
    """

    def __init__(
        self,
        vector_k: int = 3,
        max_recent_messages: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the policy.

        Args:
            vector_k: Number of semantically relevant messages to retrieve
            max_recent_messages: Number of recent messages to always include
            similarity_threshold: Minimum similarity score for relevant messages
            metadata_filter: Optional filter to apply to message metadata
        """
        self.vector_k = vector_k
        self.max_recent_messages = max_recent_messages
        self.similarity_threshold = similarity_threshold
        self.metadata_filter = metadata_filter or {}

    async def apply(self, history: List[Any]) -> List[Any]:
        """
        Apply the policy to a conversation history.

        Args:
            history: The full conversation history

        Returns:
            A combination of recent and semantically relevant messages
        """
        if not history:
            return []

        # Always include the most recent messages
        recent_messages = history[-self.max_recent_messages :] if history else []

        # If we don't have enough history for retrieval, return recent only
        if len(history) <= self.max_recent_messages:
            return recent_messages

        # Get older messages that are not in the recent set
        older_messages = history[: -self.max_recent_messages]
        if not older_messages:
            return recent_messages

        # For now, implement a simple "retrieval" that looks for keyword matches
        # In the future, this would use proper vector embeddings
        if recent_messages:
            # Use the most recent user message as the query
            user_messages = [e for e in recent_messages if e.role == "user"]
            if user_messages:
                query = user_messages[-1].content
                relevant_messages = self._retrieve_relevant(older_messages, query)
                # Combine relevant and recent messages, avoid duplicates
                result = list(relevant_messages)
                for msg in recent_messages:
                    if msg.id not in {m.id for m in result}:
                        result.append(msg)
                return result

        # If we couldn't find a good query, fall back to recent messages only
        return recent_messages

    def _retrieve_relevant(self, messages: List[Any], query: str) -> List[Any]:
        """
        Retrieve messages relevant to the query.

        This is a simple implementation that scores messages based on
        word overlap. In a real implementation, this would use vector
        similarity.

        Args:
            messages: List of messages to search
            query: Query to find relevant messages for

        Returns:
            List of relevant messages, ordered by relevance
        """
        # Simple relevance calculation based on word overlap
        query_words = set(query.lower().split())
        scored_messages = []

        for message in messages:
            # Apply metadata filter if specified
            if self.metadata_filter:
                if not all(
                    message.metadata.get(k) == v
                    for k, v in self.metadata_filter.items()
                ):
                    continue

            # Calculate simple similarity score
            message_words = set(message.content.lower().split())
            if not message_words:
                continue

            # Jaccard similarity: intersection over union
            intersection = len(query_words.intersection(message_words))
            union = len(query_words.union(message_words))
            similarity = intersection / union if union > 0 else 0

            if similarity >= self.similarity_threshold:
                scored_messages.append((similarity, message))

        # Sort by similarity score (descending) and take top k
        scored_messages.sort(reverse=True)
        return [message for _, message in scored_messages[: self.vector_k]]
