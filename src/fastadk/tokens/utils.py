"""
Utilities for token usage and cost tracking in FastADK.

This module provides helper functions to extract token usage information
from different LLM provider responses and utilities for logging and monitoring.
"""

import logging
import time
from typing import Any, Dict, Optional

import prometheus_client as prom  # type: ignore

from .models import CostCalculator, TokenBudget, TokenUsage

# Setup module-level logger
logger = logging.getLogger("fastadk.tokens")

# Define Prometheus metrics
TOKEN_COUNTER = prom.Counter(
    "fastadk_tokens_used_total",
    "Total tokens used by provider and model",
    ["provider", "model", "type"],  # Labels
)

COST_COUNTER = prom.Counter(
    "fastadk_cost_estimated_total",
    "Estimated cost in USD by provider and model",
    ["provider", "model"],  # Labels
)


def extract_token_usage_from_response(
    response: Any, provider: str, model: str
) -> Optional[TokenUsage]:
    """
    Extract token usage information from a provider response.

    This function handles different response formats from various providers
    and extracts token counts.

    Args:
        response: The response object from an LLM provider
        provider: The provider name (openai, anthropic, gemini, litellm, etc.)
        model: The model name

    Returns:
        TokenUsage object if token info was found, None otherwise
    """
    try:
        if provider.lower() == "openai":
            return _extract_openai_usage(response, model)
        elif provider.lower() == "anthropic":
            return _extract_anthropic_usage(response, model)
        elif provider.lower() == "gemini":
            return _extract_gemini_usage(response, model)
        elif provider.lower() == "litellm":
            return _extract_litellm_usage(response, model)
        else:
            logger.warning(
                "Token extraction not implemented for provider: %s", provider
            )
            return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to extract token usage: %s", str(e))
        return None


def _extract_openai_usage(response: Any, model: str) -> Optional[TokenUsage]:
    """Extract token usage from OpenAI API response."""
    try:
        # Handle different response formats
        usage = getattr(response, "usage", None)

        if usage:
            return TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                model=model,
                provider="openai",
            )

        # Try to handle dictionary response format
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model=model,
                provider="openai",
            )

        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to extract OpenAI token usage: %s", str(e))
        return None


def _extract_anthropic_usage(response: Any, model: str) -> Optional[TokenUsage]:
    """Extract token usage from Anthropic API response."""
    try:
        # Handle different response formats
        if hasattr(response, "usage"):
            usage = response.usage
            return TokenUsage(
                prompt_tokens=getattr(usage, "input_tokens", 0),
                completion_tokens=getattr(usage, "output_tokens", 0),
                model=model,
                provider="anthropic",
            )

        # Try to handle dictionary response format
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            return TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                model=model,
                provider="anthropic",
            )

        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to extract Anthropic token usage: %s", str(e))
        return None


def _extract_gemini_usage(response: Any, model: str) -> Optional[TokenUsage]:
    """Extract token usage from Google Gemini API response."""
    try:
        # Try to get usage from response
        usage = getattr(response, "usage_metadata", None)

        if usage:
            return TokenUsage(
                prompt_tokens=getattr(usage, "prompt_token_count", 0),
                completion_tokens=getattr(usage, "candidates_token_count", 0),
                model=model,
                provider="gemini",
            )

        # Try dictionary format
        if isinstance(response, dict) and "usage_metadata" in response:
            usage = response["usage_metadata"]
            return TokenUsage(
                prompt_tokens=usage.get("prompt_token_count", 0),
                completion_tokens=usage.get("candidates_token_count", 0),
                model=model,
                provider="gemini",
            )

        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to extract Gemini token usage: %s", str(e))
        return None


def _extract_litellm_usage(response: Any, model: str) -> Optional[TokenUsage]:
    """Extract token usage from LiteLLM API response."""
    try:
        # LiteLLM response format is OpenAI-compatible
        # The usage should be in the same format as OpenAI
        usage = getattr(response, "usage", None)

        if usage:
            return TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                model=model,
                provider="litellm",
            )

        # Try to handle dictionary response format
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model=model,
                provider="litellm",
            )

        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to extract LiteLLM token usage: %s", str(e))
        return None


def track_token_usage(
    usage: TokenUsage,
    budget: Optional[TokenBudget] = None,
    custom_price_per_1k: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Track token usage, update metrics, and check against budget constraints.

    Args:
        usage: The TokenUsage object to track
        budget: Optional TokenBudget to check against
        custom_price_per_1k: Optional custom pricing dictionary

    Returns:
        Dict containing tracking results, including cost and budget status
    """
    # Calculate cost
    cost = CostCalculator.calculate(usage, custom_price_per_1k)

    # Update Prometheus metrics
    TOKEN_COUNTER.labels(provider=usage.provider, model=usage.model, type="prompt").inc(
        usage.prompt_tokens
    )

    TOKEN_COUNTER.labels(
        provider=usage.provider, model=usage.model, type="completion"
    ).inc(usage.completion_tokens)

    TOKEN_COUNTER.labels(provider=usage.provider, model=usage.model, type="total").inc(
        usage.total_tokens
    )

    COST_COUNTER.labels(provider=usage.provider, model=usage.model).inc(cost)

    # Log the usage and cost
    logger.info(
        "Token usage: %s, estimated cost: $%.6f",
        usage,
        cost,
        extra={
            "event": "token_usage",
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
            "model": usage.model,
            "provider": usage.provider,
            "cost": cost,
            "timestamp": usage.timestamp,
        },
    )

    # Check against budget if provided
    budget_status = {}
    if budget:
        budget_status = budget.check_request_budget(usage, cost)

        # Log warnings if any
        for warning in budget_status.get("warnings", []):
            logger.warning(warning)

        # Log session status
        session_status = budget_status.get("session_status", {})
        if session_status.get("warnings"):
            for warning in session_status["warnings"]:
                logger.warning(warning)

    return {
        "usage": usage,
        "cost": cost,
        "timestamp": time.time(),
        "budget_status": budget_status,
    }
