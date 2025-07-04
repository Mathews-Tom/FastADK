"""
Tests for retry and circuit breaker decorators.

This module tests the retry and circuit breaker functionality provided by FastADK.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
            initial_delay=0.01,  # Use small delays for tests
            retry_on=(OperationError,),
        )(mock_func)

        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_max_attempts_reached(self):
        """Test that RetryError is raised after max attempts."""
        mock_func = AsyncMock(side_effect=OperationError("Fail"))
        decorated = retry(
            max_attempts=3, initial_delay=0.01, retry_on=(OperationError,)
        )(mock_func)

        with pytest.raises(RetryError):
            await decorated()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_timeout(self):
        """Test that timeout works correctly."""

        # Function that sleeps longer than the timeout
        async def slow_func():
            await asyncio.sleep(0.5)
            return "success"

        decorated = retry(timeout=0.1)(slow_func)

        with pytest.raises(OperationTimeoutError):
            await decorated()

    @pytest.mark.asyncio
    async def test_retry_timeout_with_multiple_attempts(self):
        """Test timeout with multiple retry attempts."""
        # First call raises error, second call succeeds but exceeds timeout
        mock_func = AsyncMock(side_effect=[OperationError("Fail"), asyncio.sleep(0.3)])

        decorated = retry(
            max_attempts=3, initial_delay=0.01, timeout=0.2, retry_on=(OperationError,)
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

        # Should only be called once since ValueError doesn't trigger retry
        mock_func.assert_called_once()

    def test_retry_sync_function(self):
        """Test retry with a synchronous function."""
        mock_func = MagicMock(side_effect=[OperationError("Fail"), "success"])
        decorated = retry(
            max_attempts=3, initial_delay=0.01, retry_on=(OperationError,)
        )(mock_func)

        result = decorated()
        assert result == "success"
        assert mock_func.call_count == 2


# Circuit Breaker tests
class TestCircuitBreaker:
    """Tests for the circuit breaker decorator."""

    @pytest.fixture(autouse=True)
    def reset_circuit_state(self):
        """Reset circuit state before each test."""
        # Access the CircuitState class via the decorator
        CircuitState = circuit_breaker().__closure__[0].cell_contents
        CircuitState.current_state = CircuitState.CLOSED
        CircuitState.failure_count = 0
        CircuitState.last_failure_time = 0
        CircuitState.last_test_time = 0

    @pytest.mark.asyncio
    async def test_circuit_closed_success(self):
        """Test successful execution with closed circuit."""
        mock_func = AsyncMock(return_value="success")
        decorated = circuit_breaker()(mock_func)

        result = await decorated()
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        # Function that always fails
        mock_func = AsyncMock(side_effect=OperationError("Service unavailable"))
        decorated = circuit_breaker(failure_threshold=2)(mock_func)

        # First two calls should attempt execution
        with pytest.raises(OperationError):
            await decorated()
        with pytest.raises(OperationError):
            await decorated()

        # Third call should raise ServiceUnavailableError without calling the function
        mock_func.reset_mock()
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await decorated()

        assert "Circuit breaker is open" in str(exc_info.value)
        # The function shouldn't be called when circuit is open
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        mock_func = AsyncMock(side_effect=OperationError("Service unavailable"))
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
                "success",  # This will be called in half-open state
            ]
        )

        decorated = circuit_breaker(failure_threshold=2, reset_timeout=0.1)(mock_func)

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

        # Next call should succeed (half-open state)
        result = await decorated()
        assert result == "success"

        # Circuit should be closed now, additional calls work normally
        mock_func.reset_mock()
        mock_func.return_value = "another success"
        result = await decorated()
        assert result == "another success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_trip_circuit(self):
        """Test that excluded exceptions don't count toward failure threshold."""
        # Mock function that raises excluded exception type
        mock_func = AsyncMock(side_effect=ValueError("Not counted"))

        decorated = circuit_breaker(failure_threshold=2, exclude=(ValueError,))(
            mock_func
        )

        # These should keep raising ValueError but not open the circuit
        for _ in range(5):  # More than the threshold
            with pytest.raises(ValueError):
                await decorated()

        # CircuitState should still be CLOSED
        CircuitState = circuit_breaker().__closure__[0].cell_contents
        assert CircuitState.current_state == CircuitState.CLOSED
        assert CircuitState.failure_count == 0

    def test_circuit_breaker_sync_function(self):
        """Test circuit breaker with synchronous function."""
        # Function that always fails
        mock_func = MagicMock(side_effect=OperationError("Service unavailable"))
        decorated = circuit_breaker(failure_threshold=2)(mock_func)

        # First two calls should attempt execution
        with pytest.raises(OperationError):
            decorated()
        with pytest.raises(OperationError):
            decorated()

        # Third call should raise ServiceUnavailableError
        mock_func.reset_mock()
        with pytest.raises(ServiceUnavailableError):
            decorated()

        # The function shouldn't be called when circuit is open
        mock_func.assert_not_called()
