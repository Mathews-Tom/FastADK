"""
Models for token usage and cost tracking in FastADK.

This module defines data classes and utilities for tracking token usage,
calculating costs, and managing token budgets.
"""

import logging
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

from .pricing import DEFAULT_PRICING

logger = logging.getLogger("fastadk.tokens")


class TokenUsage(BaseModel):
    """
    Records token usage for an LLM request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        timestamp: UNIX timestamp of when the usage was recorded
        model: Name of the model used
        provider: Provider of the model (e.g., openai, anthropic, gemini)
    """

    prompt_tokens: NonNegativeInt = Field(
        default=0, description="Tokens used in the prompt"
    )
    completion_tokens: NonNegativeInt = Field(
        default=0, description="Tokens used in the completion"
    )
    total_tokens: NonNegativeInt = Field(default=0, description="Total tokens used")
    timestamp: float = Field(
        default_factory=time.time, description="When the usage occurred"
    )
    model: str = Field(default="", description="Model name")
    provider: str = Field(
        default="", description="Provider name (openai, anthropic, gemini)"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize TokenUsage, calculating total if not provided."""
        # If total_tokens isn't provided but prompt and completion are, calculate it
        if (
            "total_tokens" not in data
            and "prompt_tokens" in data
            and "completion_tokens" in data
        ):
            data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
        super().__init__(**data)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Return a dictionary representation with additional fields."""
        result = super().dict(**kwargs)
        return result

    def __str__(self) -> str:
        """Return a string representation of the token usage."""
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}, "
            f"model={self.model}, provider={self.provider})"
        )


class CostCalculator:
    """
    Utility for calculating costs based on token usage.

    This class provides methods to calculate the cost of token usage
    based on provider-specific pricing models.
    """

    @classmethod
    def calculate(
        cls, usage: TokenUsage, custom_price_per_1k: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate the cost of token usage.

        Args:
            usage: The TokenUsage object
            custom_price_per_1k: Optional custom pricing dictionary with keys 'input' and 'output'
                                    for per 1K tokens pricing

        Returns:
            Estimated cost in USD
        """
        # Use custom pricing if provided, otherwise use default for the model
        if custom_price_per_1k is not None:
            input_price = custom_price_per_1k.get("input", 0.0)
            output_price = custom_price_per_1k.get("output", 0.0)
        else:
            # Try to get pricing for the specific model from defaults
            provider = usage.provider.lower()
            model = usage.model.lower()

            # Get default pricing if available
            provider_pricing = DEFAULT_PRICING.get(provider, {})
            model_pricing = provider_pricing.get(model, None)

            if model_pricing:
                input_price, output_price = model_pricing
            else:
                # If no specific pricing found, use conservative estimates
                logger.warning(
                    "No pricing data for %s/%s. Using conservative estimates.",
                    provider,
                    model,
                )
                input_price = 0.01  # $0.01 per 1K tokens
                output_price = 0.02  # $0.02 per 1K tokens

        # Calculate cost
        prompt_cost = (usage.prompt_tokens / 1000) * input_price
        completion_cost = (usage.completion_tokens / 1000) * output_price
        total_cost = prompt_cost + completion_cost

        return round(total_cost, 6)  # Round to 6 decimal places for currency


class TokenBudget(BaseModel):
    """
    Configurable budget for token usage and cost.

    This class provides utilities to set and check against budgets
    for token usage and estimated costs.
    """

    max_tokens_per_request: Optional[NonNegativeInt] = Field(
        default=None, description="Maximum tokens allowed per request"
    )
    max_tokens_per_session: Optional[NonNegativeInt] = Field(
        default=None, description="Maximum tokens allowed per session"
    )
    max_cost_per_request: Optional[NonNegativeFloat] = Field(
        default=None, description="Maximum cost allowed per request in USD"
    )
    max_cost_per_session: Optional[NonNegativeFloat] = Field(
        default=None, description="Maximum cost allowed per session in USD"
    )
    warn_at_percent: PositiveFloat = Field(
        default=80.0, description="Warn when budget reaches this percentage", le=100.0
    )

    # Tracking fields
    session_tokens_used: NonNegativeInt = Field(
        default=0, description="Tokens used in this session"
    )
    session_cost: NonNegativeFloat = Field(
        default=0.0, description="Cost accrued in this session"
    )

    def check_request_budget(self, usage: TokenUsage, cost: float) -> Dict[str, Any]:
        """
        Check if a request is within budget limits.

        Args:
            usage: TokenUsage for the request
            cost: Calculated cost for the request

        Returns:
            Dict with 'allowed' flag, 'warnings' list, and 'session_status'
        """
        warnings = []
        allowed = True

        # Check token budget per request
        if self.max_tokens_per_request is not None:
            if usage.total_tokens > self.max_tokens_per_request:
                warnings.append(
                    f"Token budget exceeded: {usage.total_tokens} > {self.max_tokens_per_request}"
                )
                allowed = False
            elif usage.total_tokens >= (
                self.max_tokens_per_request * self.warn_at_percent / 100
            ):
                warnings.append(
                    f"Token budget warning: {usage.total_tokens} tokens used "
                    f"({usage.total_tokens / self.max_tokens_per_request:.1%} of budget)"
                )

        # Check cost budget per request
        if self.max_cost_per_request is not None:
            if cost > self.max_cost_per_request:
                warnings.append(
                    f"Cost budget exceeded: ${cost:.6f} > ${self.max_cost_per_request:.6f}"
                )
                allowed = False
            elif cost >= (self.max_cost_per_request * self.warn_at_percent / 100):
                warnings.append(
                    f"Cost budget warning: ${cost:.6f} "
                    f"({cost / self.max_cost_per_request:.1%} of budget)"
                )

        # Update session totals if request is allowed
        if allowed:
            self.session_tokens_used += usage.total_tokens
            self.session_cost += cost

        # Check session budgets
        session_status = self._check_session_budgets()

        return {
            "allowed": allowed,
            "warnings": warnings,
            "session_status": session_status,
        }

    def _check_session_budgets(self) -> Dict[str, Any]:
        """
        Check current session against budget limits.

        Returns:
            Dict with status info including warnings and whether limits are exceeded
        """
        warnings = []
        session_limit_exceeded = False

        # Check token budget per session
        if self.max_tokens_per_session is not None:
            if self.session_tokens_used > self.max_tokens_per_session:
                warnings.append(
                    f"Session token budget exceeded: {self.session_tokens_used} > "
                    f"{self.max_tokens_per_session}"
                )
                session_limit_exceeded = True
            elif self.session_tokens_used >= (
                self.max_tokens_per_session * self.warn_at_percent / 100
            ):
                warnings.append(
                    f"Session token budget warning: {self.session_tokens_used} tokens used "
                    f"({self.session_tokens_used / self.max_tokens_per_session:.1%} of budget)"
                )

        # Check cost budget per session
        if self.max_cost_per_session is not None:
            if self.session_cost > self.max_cost_per_session:
                warnings.append(
                    f"Session cost budget exceeded: ${self.session_cost:.6f} > "
                    f"${self.max_cost_per_session:.6f}"
                )
                session_limit_exceeded = True
            elif self.session_cost >= (
                self.max_cost_per_session * self.warn_at_percent / 100
            ):
                warnings.append(
                    f"Session cost budget warning: ${self.session_cost:.6f} "
                    f"({self.session_cost / self.max_cost_per_session:.1%} of budget)"
                )

        return {
            "warnings": warnings,
            "limit_exceeded": session_limit_exceeded,
            "tokens_used": self.session_tokens_used,
            "cost_accrued": self.session_cost,
        }

    def reset_session(self) -> None:
        """Reset session counters."""
        self.session_tokens_used = 0
        self.session_cost = 0.0
