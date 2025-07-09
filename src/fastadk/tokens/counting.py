"""
Utilities for counting tokens in text for different models.

This module provides functions to count tokens in text using different
tokenizers based on the model being used.
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("fastadk.tokens")

# Global dictionary to store tokenizer instances
_TOKENIZERS: Dict[str, Any] = {}


def count_tokens(text: str, model: str) -> int:
    """
    Count tokens in a text string using the appropriate tokenizer for the model.

    Args:
        text: The text to count tokens in
        model: The model identifier (e.g., 'gpt-4.1', 'claude-3.5-sonnet', 'gemini-2.5-flash')

    Returns:
        Number of tokens in the text
    """
    tokenizer = get_tokenizer(model)
    if tokenizer is None:
        # Fall back to a simple approximation if no tokenizer is available
        return _approximate_token_count(text)

    # Count tokens using the tokenizer
    try:
        if callable(tokenizer):
            # For function-based tokenizers (Anthropic, Gemini)
            result = tokenizer(text)
            if isinstance(result, int):
                return result
            # If the result is not an int, use approximation
            return _approximate_token_count(text)
        else:
            # For object-based tokenizers (OpenAI/tiktoken)
            encoded = tokenizer.encode(text)  # type: ignore
            if hasattr(encoded, "__len__"):
                return len(encoded)
            # Fallback if encode doesn't return a sequence
            return _approximate_token_count(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("Error counting tokens with model %s: %s", model, str(e))
        return _approximate_token_count(text)


def get_tokenizer(model: str) -> Optional[Any]:
    """
    Get the appropriate tokenizer for a given model.

    Args:
        model: The model identifier

    Returns:
        A tokenizer object or function, or None if no tokenizer is available
    """
    model_lower = model.lower()

    # Check if we've already loaded this tokenizer
    if model_lower in _TOKENIZERS:
        return _TOKENIZERS[model_lower]

    # Try to load the appropriate tokenizer
    if "gpt" in model_lower or "text-embedding" in model_lower:
        tokenizer = _load_tiktoken_tokenizer(model_lower)
    elif "claude" in model_lower:
        tokenizer = _load_anthropic_tokenizer()
    elif "gemini" in model_lower:
        tokenizer = _load_gemini_tokenizer()
    else:
        # No specific tokenizer for this model
        logger.warning("No tokenizer available for model: %s", model)
        tokenizer = None

    # Cache the tokenizer
    _TOKENIZERS[model_lower] = tokenizer
    return tokenizer


def _load_tiktoken_tokenizer(model: str) -> Optional[Any]:
    """Load a tiktoken tokenizer for OpenAI models."""
    try:
        import tiktoken

        # Try to get the encoding for the specific model
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for newer models
            logger.warning(
                "Specific tokenizer not found for %s, using cl100k_base", model
            )
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning("tiktoken not installed. Install with: uv add tiktoken")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Error loading tiktoken tokenizer: %s", str(e))
        return None


def _load_anthropic_tokenizer() -> Optional[Callable[[str], int]]:
    """Load a tokenizer for Anthropic models."""
    try:
        import anthropic

        # Define a safe tokenizer function
        def count_tokens_anthropic(text: str) -> int:
            try:
                # Try to use the count_tokens method if it exists
                client = anthropic.Anthropic()
                # We need to handle this dynamically because different versions have different APIs
                if hasattr(client, "count_tokens"):
                    # For static type checking we need to ignore this
                    result = client.count_tokens(text)  # type: ignore
                    if isinstance(result, int):
                        return result
            except (AttributeError, TypeError):
                pass
            # Fallback
            return _approximate_token_count(text)

        return count_tokens_anthropic
    except ImportError:
        logger.warning("anthropic not installed. Install with: uv add anthropic")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Error loading Anthropic tokenizer: %s", str(e))
        return None


def _load_gemini_tokenizer() -> Optional[Callable[[str], int]]:
    """Load a tokenizer for Google Gemini models."""
    try:
        import google.generativeai as genai

        # Define a safe tokenizer function
        def count_tokens_gemini(text: str) -> int:
            try:
                # Try to use the count_tokens method if it exists
                if hasattr(genai, "count_tokens"):
                    # For static type checking we need to ignore this
                    result = genai.count_tokens(text)  # type: ignore
                    if hasattr(result, "total_tokens"):
                        total = result.total_tokens
                        if isinstance(total, int):
                            return total
            except (AttributeError, TypeError):
                pass
            # Fallback
            return _approximate_token_count(text)

        return count_tokens_gemini
    except ImportError:
        logger.warning("google-generativeai not installed or outdated")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Error loading Gemini tokenizer: %s", str(e))
        return None


def _approximate_token_count(text: str) -> int:
    """
    Approximate token count based on simple heuristics.

    This is a fallback method when no tokenizer is available.
    It's not accurate but provides a rough estimate.

    Args:
        text: The text to count tokens in

    Returns:
        Approximate token count
    """
    # Very rough approximation: ~4 characters per token
    return max(1, len(text) // 4)


def estimate_tokens_and_cost(
    text: str, model: str, is_completion: bool = False
) -> Dict[str, Any]:
    """
    Estimate tokens and cost for a text string.

    Args:
        text: The text to estimate tokens for
        model: The model identifier
        is_completion: Whether this text is a completion (affects pricing)

    Returns:
        Dictionary with token count and estimated cost
    """
    from .models import CostCalculator, TokenUsage

    # Count tokens
    token_count = count_tokens(text, model)

    # Create a TokenUsage object
    if is_completion:
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=token_count,
            model=model,
            provider=_get_provider_from_model(model),
        )
    else:
        usage = TokenUsage(
            prompt_tokens=token_count,
            completion_tokens=0,
            model=model,
            provider=_get_provider_from_model(model),
        )

    # Calculate cost
    cost = CostCalculator.calculate(usage)

    return {
        "tokens": token_count,
        "cost": cost,
        "provider": usage.provider,
        "model": model,
    }


def _get_provider_from_model(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()

    if (
        "gpt" in model_lower
        or model_lower.startswith("text-")
        or "davinci" in model_lower
    ):
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "gemini"
    else:
        return "unknown"
