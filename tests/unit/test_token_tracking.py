"""
Tests for token tracking and cost estimation features.

This module contains unit tests for TokenUsage, CostCalculator, and TokenBudget classes.
"""

from unittest.mock import MagicMock, patch

import pytest

from fastadk.tokens.models import CostCalculator, TokenBudget, TokenUsage
from fastadk.tokens.utils import extract_token_usage_from_response, track_token_usage


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_init_with_token_counts(self):
        """Test creating TokenUsage with explicit token counts."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30  # Should auto-calculate

    def test_init_with_total_override(self):
        """Test creating TokenUsage with total tokens explicitly provided."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=35)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 35  # Should use provided value, not calculate

    def test_string_representation(self):
        """Test string representation of TokenUsage."""
        usage = TokenUsage(
            prompt_tokens=10, completion_tokens=20, model="test-model", provider="test"
        )
        str_repr = str(usage)
        assert "prompt=10" in str_repr
        assert "completion=20" in str_repr
        assert "total=30" in str_repr
        assert "model=test-model" in str_repr
        assert "provider=test" in str_repr

    def test_dict_method(self):
        """Test dict representation of TokenUsage."""
        usage = TokenUsage(
            prompt_tokens=10, completion_tokens=20, model="test-model", provider="test"
        )
        data = usage.dict()
        assert data["prompt_tokens"] == 10
        assert data["completion_tokens"] == 20
        assert data["total_tokens"] == 30
        assert data["model"] == "test-model"
        assert data["provider"] == "test"


class TestCostCalculator:
    """Tests for CostCalculator class."""

    def test_calculate_with_default_pricing(self):
        """Test cost calculation using default pricing."""
        # Create a usage object for GPT-4
        usage = TokenUsage(
            prompt_tokens=1000, completion_tokens=500, model="gpt-4", provider="openai"
        )

        # Calculate cost
        cost = CostCalculator.calculate(usage)

        # Using pricing for GPT-4: $0.03/1K input, $0.06/1K output
        expected_cost = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_calculate_with_custom_pricing(self):
        """Test cost calculation using custom pricing."""
        # Create a usage object
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="custom-model",
            provider="custom",
        )

        # Define custom pricing
        custom_pricing = {
            "input": 0.01,  # $0.01 per 1K input tokens
            "output": 0.02,  # $0.02 per 1K output tokens
        }

        # Calculate cost
        cost = CostCalculator.calculate(usage, custom_pricing)

        # Expected cost calculation
        expected_cost = (1000 / 1000 * 0.01) + (500 / 1000 * 0.02)
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_fallback_pricing_for_unknown_model(self):
        """Test that a fallback price is used for unknown models."""
        # Create a usage object for an unknown model
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="unknown-model",
            provider="unknown",
        )

        # Calculate cost
        with patch("fastadk.tokens.models.logger") as mock_logger:
            cost = CostCalculator.calculate(usage)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "No pricing data for" in mock_logger.warning.call_args[0][0]

        # Cost should still be calculated using fallback values
        assert cost > 0


class TestTokenBudget:
    """Tests for TokenBudget class."""

    def test_check_request_within_budget(self):
        """Test request within budget limits."""
        # Create a budget with limits
        budget = TokenBudget(max_tokens_per_request=1000, max_cost_per_request=0.05)

        # Create a usage within limits
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)
        cost = 0.03

        # Check budget
        result = budget.check_request_budget(usage, cost)

        # Should be allowed
        assert result["allowed"] is True
        assert len(result["warnings"]) == 0

        # Session should be updated
        assert budget.session_tokens_used == 600
        assert budget.session_cost == 0.03

    def test_check_request_exceeds_token_budget(self):
        """Test request exceeding token budget."""
        # Create a budget with limits
        budget = TokenBudget(max_tokens_per_request=500, max_cost_per_request=0.05)

        # Create a usage exceeding token limit
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)  # 600 total
        cost = 0.03

        # Check budget
        result = budget.check_request_budget(usage, cost)

        # Should not be allowed
        assert result["allowed"] is False
        assert len(result["warnings"]) > 0
        assert "Token budget exceeded" in result["warnings"][0]

        # Session should not be updated
        assert budget.session_tokens_used == 0
        assert budget.session_cost == 0.0

    def test_check_request_exceeds_cost_budget(self):
        """Test request exceeding cost budget."""
        # Create a budget with limits
        budget = TokenBudget(max_tokens_per_request=1000, max_cost_per_request=0.02)

        # Create a usage exceeding cost limit
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)
        cost = 0.03

        # Check budget
        result = budget.check_request_budget(usage, cost)

        # Should not be allowed
        assert result["allowed"] is False
        assert len(result["warnings"]) > 0
        assert "Cost budget exceeded" in result["warnings"][0]

        # Session should not be updated
        assert budget.session_tokens_used == 0
        assert budget.session_cost == 0.0

    def test_warning_at_percentage_threshold(self):
        """Test warning when approaching budget threshold."""
        # Create a budget with limits and 80% warning threshold
        budget = TokenBudget(
            max_tokens_per_request=1000, max_cost_per_request=0.05, warn_at_percent=80.0
        )

        # Create a usage at 85% of token limit
        usage = TokenUsage(prompt_tokens=700, completion_tokens=150)  # 850 total
        cost = 0.03

        # Check budget
        result = budget.check_request_budget(usage, cost)

        # Should be allowed but with warning
        assert result["allowed"] is True
        assert len(result["warnings"]) > 0
        assert "Token budget warning" in result["warnings"][0]

        # Session should be updated
        assert budget.session_tokens_used == 850
        assert budget.session_cost == 0.03

    def test_session_budget_tracking(self):
        """Test session budget tracking over multiple requests."""
        # Create a budget with session limits
        budget = TokenBudget(max_tokens_per_session=2000, max_cost_per_session=0.10)

        # First request
        usage1 = TokenUsage(prompt_tokens=400, completion_tokens=200)
        cost1 = 0.03
        result1 = budget.check_request_budget(usage1, cost1)

        assert result1["allowed"] is True
        assert budget.session_tokens_used == 600
        assert budget.session_cost == 0.03

        # Second request
        usage2 = TokenUsage(prompt_tokens=500, completion_tokens=300)
        cost2 = 0.04
        result2 = budget.check_request_budget(usage2, cost2)

        assert result2["allowed"] is True
        assert budget.session_tokens_used == 1400  # 600 + 800
        assert budget.session_cost == 0.07  # 0.03 + 0.04

        # Third request that exceeds session limit
        usage3 = TokenUsage(prompt_tokens=700, completion_tokens=400)
        cost3 = 0.05
        result3 = budget.check_request_budget(usage3, cost3)

        # Should be allowed for the request (no per-request limit)
        # But session status should indicate limit exceeded
        assert result3["allowed"] is True
        assert budget.session_tokens_used == 2500  # 1400 + 1100
        assert budget.session_cost == pytest.approx(0.12, rel=1e-6)  # 0.07 + 0.05

        # Session status should have warnings
        session_status = result3["session_status"]
        assert session_status["limit_exceeded"] is True
        assert len(session_status["warnings"]) > 0
        assert "Session token budget exceeded" in session_status["warnings"][0]

    def test_reset_session(self):
        """Test resetting session counters."""
        # Create a budget
        budget = TokenBudget(max_tokens_per_session=1000, max_cost_per_session=0.05)

        # Add some usage
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)
        cost = 0.03
        budget.check_request_budget(usage, cost)

        # Verify session is updated
        assert budget.session_tokens_used == 600
        assert budget.session_cost == 0.03

        # Reset session
        budget.reset_session()

        # Verify counters are reset
        assert budget.session_tokens_used == 0
        assert budget.session_cost == 0.0


class TestTokenUtilities:
    """Tests for token utilities."""

    def test_extract_openai_usage(self):
        """Test extracting token usage from OpenAI response."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        # Extract usage
        usage = extract_token_usage_from_response(mock_response, "openai", "gpt-4")

        # Verify usage
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.provider == "openai"
        assert usage.model == "gpt-4"

    def test_extract_anthropic_usage(self):
        """Test extracting token usage from Anthropic response."""
        # Mock Anthropic response
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        # Extract usage
        usage = extract_token_usage_from_response(
            mock_response, "anthropic", "claude-3-opus"
        )

        # Verify usage
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150  # Should be calculated
        assert usage.provider == "anthropic"
        assert usage.model == "claude-3-opus"

    def test_extract_gemini_usage(self):
        """Test extracting token usage from Gemini response."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        # Extract usage
        usage = extract_token_usage_from_response(
            mock_response, "gemini", "gemini-1.5-pro"
        )

        # Verify usage
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150  # Should be calculated
        assert usage.provider == "gemini"
        assert usage.model == "gemini-1.5-pro"

    def test_extract_litellm_usage(self):
        """Test extracting token usage from LiteLLM response."""
        # Mock LiteLLM response (uses OpenAI-compatible format)
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        # Extract usage
        usage = extract_token_usage_from_response(mock_response, "litellm", "gpt-4")

        # Verify usage
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.provider == "litellm"
        assert usage.model == "gpt-4"

    def test_extract_fallback_for_unknown_provider(self):
        """Test fallback when provider is unknown."""
        # Mock response
        mock_response = MagicMock()

        # Extract usage for unknown provider
        with patch("fastadk.tokens.utils.logger") as mock_logger:
            usage = extract_token_usage_from_response(
                mock_response, "unknown", "unknown-model"
            )

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "not implemented for provider" in mock_logger.warning.call_args[0][0]

        # Should return None
        assert usage is None

    def test_track_token_usage_with_metrics(self):
        """Test tracking token usage with metrics."""
        # Create usage
        usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, model="test-model", provider="test"
        )

        # Mock Prometheus counters
        with (
            patch("fastadk.tokens.utils.TOKEN_COUNTER") as mock_token_counter,
            patch("fastadk.tokens.utils.COST_COUNTER") as mock_cost_counter,
            patch("fastadk.tokens.utils.logger") as mock_logger,
        ):
            # Setup mock counter labels
            mock_token_prompt = MagicMock()
            mock_token_completion = MagicMock()
            mock_token_total = MagicMock()
            mock_cost = MagicMock()

            mock_token_counter.labels.side_effect = [
                mock_token_prompt,
                mock_token_completion,
                mock_token_total,
            ]
            mock_cost_counter.labels.return_value = mock_cost

            # Track usage
            result = track_token_usage(usage)

            # Verify metrics were incremented
            mock_token_prompt.inc.assert_called_once_with(100)
            mock_token_completion.inc.assert_called_once_with(50)
            mock_token_total.inc.assert_called_once_with(150)
            mock_cost.inc.assert_called_once()

            # Verify logging
            mock_logger.info.assert_called_once()
            assert "Token usage" in mock_logger.info.call_args[0][0]

            # Verify result
            assert result["usage"] == usage
            assert "cost" in result
            assert "timestamp" in result

    def test_track_token_usage_with_budget(self):
        """Test tracking token usage with budget constraints."""
        # Create usage
        usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, model="test-model", provider="test"
        )

        # Create budget
        budget = TokenBudget(max_tokens_per_request=200, max_cost_per_request=0.05)

        # Mock Prometheus counters
        with (
            patch("fastadk.tokens.utils.TOKEN_COUNTER") as mock_token_counter,
            patch("fastadk.tokens.utils.COST_COUNTER") as mock_cost_counter,
            patch("fastadk.tokens.utils.logger"),
        ):
            mock_token_counter.labels.return_value = MagicMock()
            mock_cost_counter.labels.return_value = MagicMock()

            # Track usage
            result = track_token_usage(usage, budget)

            # Verify budget status
            assert "budget_status" in result
            assert result["budget_status"]["allowed"] is True

            # Verify budget was updated
            assert budget.session_tokens_used == 150
            assert budget.session_cost > 0
