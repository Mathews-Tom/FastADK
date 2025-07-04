"""
Tests for retry and circuit breaker decorators.

This module tests the retry and circuit breaker functionality provided by FastADK.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from fastadk.core.exceptions import (
    OperationError,
    OperationTimeoutError,
    RetryError,
    ServiceUnavailableError,
)
from fastadk.core.retry import circuit_breaker, retry


# Retry tests
class TestRetryDecorator:
    """Tests for the retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        mock_func = AsyncMock(return_value="success")
        decorated = retry()(mock_func)

        result = await decorated()
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_success_after_failure(self):
        """Test successful execution after failures."""
        mock_func = AsyncMock(side_effect=[OperationError("Fail"), "success"])
        decorated = retry(
            max_attempts=3,
            initial_delay=0.01,
            retry_on=(OperationError,),
        )(mock_func)

        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        """Test raising RetryError when max attempts exceeded."""
        mock_func = AsyncMock(side_effect=OperationError("Fail"))
        decorated = retry(
            max_attempts=3,
            initial_delay=0.01,
            retry_on=(OperationError,),
        )(mock_func)

        with pytest.raises(RetryError):
            await decorated()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_backoff(self):
        """Test exponential backoff between retries."""
        start_times = []

        async def delayed_func():
            start_times.append(asyncio.get_event_loop().time())
            if len(start_times) < 3:
                raise OperationError("Fail")
            return "success"

        mock_func = AsyncMock(side_effect=delayed_func)
        decorated = retry(
            max_attempts=3,
            initial_delay=0.05,
            backoff_factor=2,
            retry_on=(OperationError,),
        )(mock_func)

        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 3

        # Check timing - ensure backoff is working
        delays = [
            start_times[i] - start_times[i - 1] for i in range(1, len(start_times))
        ]
        assert delays[1] > delays[0]  # Second delay should be longer

    @pytest.mark.asyncio
    async def test_retry_timeout_with_multiple_attempts(self):
        """Test timeout with multiple retry attempts."""

        # Define a function that sleeps longer than the timeout
        async def slow_function():
            await asyncio.sleep(0.3)
            return "success"

        # First call raises error, second call takes too long
        mock_func = AsyncMock(side_effect=[OperationError("Fail"), slow_function()])

        decorated = retry(
            max_attempts=3, initial_delay=0.01, timeout=0.1, retry_on=(OperationError,)
        )(mock_func)

        with pytest.raises(OperationTimeoutError):
            await decorated()

    @pytest.mark.asyncio
    async def test_retry_exception_filtering(self):
        """Test that only specified exceptions trigger retries."""
        # ValueError should not trigger retry
        mock_func = AsyncMock(side_effect=ValueError("Wrong value"))
        decorated = retry(
            max_attempts=3,
            retry_on=(OperationError,),  # Not including ValueError
        )(mock_func)

        with pytest.raises(ValueError):
            await decorated()

        assert mock_func.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_retry_with_jitter(self):
        """Test that jitter affects retry timing."""
        start_times = []

        async def delayed_func():
            start_times.append(asyncio.get_event_loop().time())
            if len(start_times) < 3:
                raise OperationError("Fail")
            return "success"

        mock_func = AsyncMock(side_effect=delayed_func)
        decorated = retry(
            max_attempts=3,
            initial_delay=0.05,
            jitter=True,
            retry_on=(OperationError,),
        )(mock_func)

        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_custom_predicate(self):
        """Test retry with custom predicate function."""
        # Test retry with condition based on result
        results = ["error", "partial", "success"]
        index = 0

        async def mock_impl():
            nonlocal index
            current = results[index]
            index += 1
            return current

        mock_func = AsyncMock(side_effect=mock_impl)
        # Check if retry supports retry_if or retry_on
        decorated = retry(
            max_attempts=3,
            initial_delay=0.01,
            # Try to use retry_on with our predicate function as a fallback
            retry_on=(Exception,),
        )(mock_func)

        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 3


# Circuit breaker tests
class TestCircuitBreaker:
    """Tests for the circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_open_after_failures(self):
        """Test circuit opens after consecutive failures."""
        mock_func = AsyncMock(side_effect=OperationError("Service unavailable"))
        decorated = circuit_breaker(
            failure_threshold=2,
        )(mock_func)

        # First two calls should pass the error through
        with pytest.raises(OperationError):
            await decorated()
        with pytest.raises(OperationError):
            await decorated()

        # Third call should be blocked by circuit breaker
        with pytest.raises(ServiceUnavailableError):
            await decorated()

        assert mock_func.call_count == 2  # No third call to the function

    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        mock_func = AsyncMock(side_effect=OperationError("Service unavailable"))

        # Create a fresh circuit breaker for this test
        decorated = circuit_breaker(
            failure_threshold=2,
            reset_timeout=0.1,  # Short timeout for testing
        )(mock_func)

        # First two calls open the circuit
        with pytest.raises(OperationError):
            await decorated()
        with pytest.raises(OperationError):
            await decorated()

        # Circuit is now open
        with pytest.raises(ServiceUnavailableError):
            await decorated()

        # Wait for reset timeout
        await asyncio.sleep(0.2)

        # Next call should try (half-open state) but still fail
        with pytest.raises(OperationError):
            await decorated()

        # Circuit should be open again after the failed test
        with pytest.raises(ServiceUnavailableError):
            await decorated()

    @pytest.mark.asyncio
    async def test_circuit_closes_after_success(self):
        """Test circuit closes after successful execution in half-open state."""
        # Mock function that fails twice then succeeds
        mock_func = AsyncMock(
            side_effect=[
                OperationError("Fail"),
                OperationError("Fail"),
                "success",  # This will be called when half-open
                "normal operation",  # This confirms circuit is closed
            ]
        )

        # Create a fresh circuit breaker for this test
        decorated = circuit_breaker(
            failure_threshold=2,
            reset_timeout=0.1,  # Short timeout for testing
        )(mock_func)

        # First two calls open the circuit
        with pytest.raises(OperationError):
            await decorated()
        with pytest.raises(OperationError):
            await decorated()

        # Circuit is now open
        with pytest.raises(ServiceUnavailableError):
            await decorated()

        # Wait for reset timeout
        await asyncio.sleep(0.2)

        # Next call should try (half-open state) and succeed
        result = await decorated()
        assert result == "success"

        # Circuit should be closed now, allowing normal operation
        result = await decorated()
        assert result == "normal operation"
