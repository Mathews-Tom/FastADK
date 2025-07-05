"""
Retry decorator for FastADK.

This module provides a retry mechanism for functions and methods that might fail
temporarily and should be retried with backoff.
"""

import asyncio
import functools
import secrets
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast

from loguru import logger

from fastadk.core.exceptions import (
    OperationError,
    OperationTimeoutError,
    RetryError,
    ServiceUnavailableError,
)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Iterable[type[Exception]] = (
        OperationError,
        ServiceUnavailableError,
    ),
    timeout: float | None = None,
) -> Callable[[F], F]:
    """
    Retry decorator for functions that might fail temporarily.

    This decorator implements exponential backoff with optional jitter for retrying
    functions that might fail due to temporary issues like network problems.

    Args:
        max_attempts: Maximum number of attempts (including the first one)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delay
        retry_on: Exception types that should trigger a retry
        timeout: Overall timeout in seconds for all attempts

    Returns:
        A decorator function
    """

    def decorator(func: F) -> F:
        """Decorator function for retry logic."""

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for async functions."""
            start_time = time.time()
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    # Apply timeout if specified
                    if timeout is not None:
                        # Calculate remaining time for this attempt
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise OperationTimeoutError(
                                message=f"Operation timed out after {elapsed:.2f}s",
                                error_code="OPERATION_TIMEOUT",
                                details={
                                    "timeout": timeout,
                                    "elapsed": elapsed,
                                    "attempts": attempt,
                                },
                            )

                        # Run with timeout
                        return await asyncio.wait_for(
                            func(*args, **kwargs), timeout=remaining
                        )
                    else:
                        # Run without timeout
                        return await func(*args, **kwargs)

                except asyncio.TimeoutError as exc:
                    elapsed = time.time() - start_time
                    raise OperationTimeoutError(
                        message=f"Operation timed out after {elapsed:.2f}s",
                        error_code="OPERATION_TIMEOUT",
                        details={
                            "timeout": timeout,
                            "elapsed": elapsed,
                            "attempts": attempt + 1,
                        },
                    ) from exc
                except retry_on as e:
                    last_exception = e
                    # Don't retry if this is the last attempt
                    if attempt >= max_attempts - 1:
                        break

                    # Calculate delay for next attempt
                    if jitter:
                        # Use cryptographically secure random numbers
                        jitter_amount = secrets.SystemRandom().random() * delay * 0.2
                        current_delay = delay + jitter_amount
                    else:
                        current_delay = delay

                    # Check if we'll exceed the timeout with the next delay
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed + current_delay >= timeout:
                            remaining = timeout - elapsed
                            if remaining > 0:
                                # Sleep for the remaining time before timeout
                                logger.debug(
                                    f"Attempt {attempt + 1}/{max_attempts} failed, "
                                    f"will retry in {remaining:.2f}s (timeout limit)"
                                )
                                await asyncio.sleep(remaining)
                            break  # This will exit the loop and raise RetryError

                    # Log and wait
                    logger.debug(
                        f"Attempt {attempt + 1}/{max_attempts} failed with {type(e).__name__}, "
                        f"retrying in {current_delay:.2f}s"
                    )
                    await asyncio.sleep(current_delay)

                    # Increase delay for next attempt using exponential backoff
                    delay = min(delay * backoff_factor, max_delay)

            # If we've exhausted all retries, raise a RetryError
            if last_exception is not None:
                elapsed = time.time() - start_time
                raise RetryError(
                    message=f"Operation failed after {max_attempts} attempts ({elapsed:.2f}s)",
                    error_code="RETRY_EXHAUSTED",
                    details={
                        "attempts": max_attempts,
                        "elapsed": elapsed,
                        "original_error": str(last_exception),
                        "original_error_type": type(last_exception).__name__,
                    },
                ) from last_exception

            # This should never happen, but makes the type checker happy
            raise RuntimeError("Unexpected code path in retry decorator")

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for synchronous functions."""
            start_time = time.time()
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    # Apply timeout if specified (only for async functions)
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise OperationTimeoutError(
                                message=f"Operation timed out after {elapsed:.2f}s",
                                error_code="OPERATION_TIMEOUT",
                                details={
                                    "timeout": timeout,
                                    "elapsed": elapsed,
                                    "attempts": attempt,
                                },
                            )

                    # Run the function
                    return func(*args, **kwargs)

                except retry_on as e:
                    last_exception = e
                    # Don't retry if this is the last attempt
                    if attempt >= max_attempts - 1:
                        break

                    # Calculate delay for next attempt
                    if jitter:
                        # Use cryptographically secure random numbers
                        jitter_amount = secrets.SystemRandom().random() * delay * 0.2
                        current_delay = delay + jitter_amount
                    else:
                        current_delay = delay

                    # Check if we'll exceed the timeout with the next delay
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed + current_delay >= timeout:
                            remaining = timeout - elapsed
                            if remaining > 0:
                                # Sleep for the remaining time before timeout
                                logger.debug(
                                    f"Attempt {attempt + 1}/{max_attempts} failed, "
                                    f"will retry in {remaining:.2f}s (timeout limit)"
                                )
                                time.sleep(remaining)
                            break  # This will exit the loop and raise RetryError

                    # Log and wait
                    logger.debug(
                        f"Attempt {attempt + 1}/{max_attempts} failed with {type(e).__name__}, "
                        f"retrying in {current_delay:.2f}s"
                    )
                    time.sleep(current_delay)

                    # Increase delay for next attempt using exponential backoff
                    delay = min(delay * backoff_factor, max_delay)

            # If we've exhausted all retries, raise a RetryError
            if last_exception is not None:
                elapsed = time.time() - start_time
                raise RetryError(
                    message=f"Operation failed after {max_attempts} attempts ({elapsed:.2f}s)",
                    error_code="RETRY_EXHAUSTED",
                    details={
                        "attempts": max_attempts,
                        "elapsed": elapsed,
                        "original_error": str(last_exception),
                        "original_error_type": type(last_exception).__name__,
                    },
                ) from last_exception

            # This should never happen, but makes the type checker happy
            raise RuntimeError("Unexpected code path in retry decorator")

        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    half_open_timeout: float = 5.0,
    exclude: Iterable[type[Exception]] = (),
) -> Callable[[F], F]:
    """
    Circuit breaker pattern implementation.

    This decorator implements the circuit breaker pattern to prevent repeated calls
    to a service that is failing, allowing it time to recover.

    Args:
        failure_threshold: Number of failures before opening the circuit
        reset_timeout: Time in seconds to wait before attempting to reset the circuit
        half_open_timeout: Time in seconds to wait for a test request in half-open state
        exclude: Exception types that should not count as failures

    Returns:
        A decorator function
    """

    # Shared state for the circuit breaker
    # Using class attributes to maintain state across invocations
    class CircuitState:
        # Possible states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
        CLOSED = "CLOSED"
        OPEN = "OPEN"
        HALF_OPEN = "HALF_OPEN"

        current_state: str = CLOSED
        failure_count: int = 0
        last_failure_time: float = 0
        last_test_time: float = 0

    def decorator(func: F) -> F:
        """Decorator function for circuit breaker logic."""

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for async functions."""
            now = time.time()

            # Check if the circuit is OPEN
            if CircuitState.current_state == CircuitState.OPEN:
                elapsed = now - CircuitState.last_failure_time
                if elapsed >= reset_timeout:
                    # Transition to HALF_OPEN to test the service
                    CircuitState.current_state = CircuitState.HALF_OPEN
                    CircuitState.last_test_time = now
                    logger.info(
                        f"Circuit transitioning from OPEN to HALF_OPEN after {elapsed:.2f}s"
                    )
                else:
                    # Circuit is OPEN and timeout hasn't expired
                    raise ServiceUnavailableError(
                        message="Circuit breaker is open",
                        error_code="CIRCUIT_OPEN",
                        details={
                            "failures": CircuitState.failure_count,
                            "seconds_remaining": reset_timeout - elapsed,
                        },
                    )

            # If circuit is HALF_OPEN, only allow one test request
            if CircuitState.current_state == CircuitState.HALF_OPEN:
                elapsed = now - CircuitState.last_test_time
                if elapsed < half_open_timeout:
                    # Only one test request allowed in half_open_timeout period
                    raise ServiceUnavailableError(
                        message="Circuit breaker is recovering",
                        error_code="CIRCUIT_RECOVERING",
                        details={
                            "seconds_remaining": half_open_timeout - elapsed,
                        },
                    )
                # Update the test time for the current attempt
                CircuitState.last_test_time = now

            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Success: If the circuit was HALF_OPEN, close it
                if CircuitState.current_state != CircuitState.CLOSED:
                    logger.info(
                        "Circuit transitioned to CLOSED after successful execution"
                    )
                    CircuitState.current_state = CircuitState.CLOSED
                    CircuitState.failure_count = 0

                return result

            except exclude:
                # Don't count excluded exceptions as failures
                raise

            except Exception:
                # Increment failure count
                CircuitState.failure_count += 1
                CircuitState.last_failure_time = time.time()

                # Check if we need to open the circuit
                if (
                    CircuitState.current_state == CircuitState.CLOSED
                    and CircuitState.failure_count >= failure_threshold
                ):
                    CircuitState.current_state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit OPENED after {CircuitState.failure_count} failures"
                    )

                # If in HALF_OPEN state and a failure occurs, go back to OPEN
                if CircuitState.current_state == CircuitState.HALF_OPEN:
                    CircuitState.current_state = CircuitState.OPEN
                    logger.warning("Circuit REOPENED after test request failed")

                # Re-raise the original exception
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for synchronous functions."""
            now = time.time()

            # Check if the circuit is OPEN
            if CircuitState.current_state == CircuitState.OPEN:
                elapsed = now - CircuitState.last_failure_time
                if elapsed >= reset_timeout:
                    # Transition to HALF_OPEN to test the service
                    CircuitState.current_state = CircuitState.HALF_OPEN
                    CircuitState.last_test_time = now
                    logger.info(
                        f"Circuit transitioning from OPEN to HALF_OPEN after {elapsed:.2f}s"
                    )
                else:
                    # Circuit is OPEN and timeout hasn't expired
                    raise ServiceUnavailableError(
                        message="Circuit breaker is open",
                        error_code="CIRCUIT_OPEN",
                        details={
                            "failures": CircuitState.failure_count,
                            "seconds_remaining": reset_timeout - elapsed,
                        },
                    )

            # If circuit is HALF_OPEN, only allow one test request
            if CircuitState.current_state == CircuitState.HALF_OPEN:
                elapsed = now - CircuitState.last_test_time
                if elapsed < half_open_timeout:
                    # Only one test request allowed in half_open_timeout period
                    raise ServiceUnavailableError(
                        message="Circuit breaker is recovering",
                        error_code="CIRCUIT_RECOVERING",
                        details={
                            "seconds_remaining": half_open_timeout - elapsed,
                        },
                    )
                # Update the test time for the current attempt
                CircuitState.last_test_time = now

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Success: If the circuit was HALF_OPEN, close it
                if CircuitState.current_state != CircuitState.CLOSED:
                    logger.info(
                        "Circuit transitioned to CLOSED after successful execution"
                    )
                    CircuitState.current_state = CircuitState.CLOSED
                    CircuitState.failure_count = 0

                return result

            except exclude:
                # Don't count excluded exceptions as failures
                raise

            except Exception:
                # Increment failure count
                CircuitState.failure_count += 1
                CircuitState.last_failure_time = time.time()

                # Check if we need to open the circuit
                if (
                    CircuitState.current_state == CircuitState.CLOSED
                    and CircuitState.failure_count >= failure_threshold
                ):
                    CircuitState.current_state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit OPENED after {CircuitState.failure_count} failures"
                    )

                # If in HALF_OPEN state and a failure occurs, go back to OPEN
                if CircuitState.current_state == CircuitState.HALF_OPEN:
                    CircuitState.current_state = CircuitState.OPEN
                    logger.warning("Circuit REOPENED after test request failed")

                # Re-raise the original exception
                raise

        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator
