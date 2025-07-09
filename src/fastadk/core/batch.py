"""
Batch processing utilities for FastADK.

This module provides tools for processing multiple items in parallel or in
batches, with support for rate limiting, error handling, and progress tracking.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, cast

from .workflow import Workflow

# Type variables for batch input and output
T = TypeVar("T")
U = TypeVar("U")

logger = logging.getLogger("fastadk.batch")


@dataclass
class BatchResult(Generic[T, U]):
    """Result of a batch operation."""

    successful: list[tuple[T, U]] = field(default_factory=list)
    failed: list[tuple[T, Exception]] = field(default_factory=list)
    skipped: list[T] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def success_count(self) -> int:
        """Number of successful operations."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed operations."""
        return len(self.failed)

    @property
    def skip_count(self) -> int:
        """Number of skipped operations."""
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        """Total number of operations."""
        return self.success_count + self.failure_count + self.skip_count


class BatchUtils:
    """
    Utility class for batch processing operations.

    This class provides methods for processing items in parallel, with control
    over concurrency, rate limiting, and error handling.
    """

    @staticmethod
    async def process_parallel(
        items: Iterable[T],
        process_func: Callable[[T], U],
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
        skip_errors: bool = False,
    ) -> BatchResult[T, U]:
        """
        Process multiple items in parallel.

        Args:
            items: Items to process
            process_func: Function to apply to each item
            max_concurrency: Maximum number of concurrent operations
            timeout: Optional timeout per item in seconds
            skip_errors: Whether to continue processing on errors

        Returns:
            BatchResult containing successful and failed operations
        """
        result = BatchResult[T, U]()
        start_time = time.time()

        # Convert items to a list if it's not already
        items_list = list(items)
        total_items = len(items_list)

        if total_items == 0:
            logger.warning("No items to process")
            result.total_time = 0.0
            return result

        logger.info(
            "Processing %s items in parallel (max concurrency: %s)",
            total_items,
            max_concurrency,
        )

        # Use a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(item: T) -> tuple[T, U, Optional[Exception]]:
            """Process a single item with error handling."""
            async with semaphore:
                try:
                    # Handle both async and sync functions
                    if asyncio.iscoroutinefunction(process_func):
                        if timeout:
                            result = await asyncio.wait_for(process_func(item), timeout)
                        else:
                            result = await process_func(item)
                    else:
                        if timeout:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(lambda: process_func(item)), timeout
                            )
                        else:
                            result = await asyncio.to_thread(lambda: process_func(item))

                    return item, result, None
                except asyncio.TimeoutError as timeout_err:
                    logger.error("Timeout processing item %s", item)
                    return item, cast(U, None), timeout_err
                except Exception as e:
                    logger.error("Error processing item %s: %s", item, e)
                    return item, cast(U, None), e

        # Create tasks for all items
        tasks = [asyncio.create_task(process_item(item)) for item in items_list]

        # Process results as they complete
        for task in asyncio.as_completed(tasks):
            item, item_result, error = await task
            if error:
                if skip_errors:
                    result.failed.append((item, error))
                else:
                    # Cancel remaining tasks
                    for t in tasks:
                        if not t.done():
                            t.cancel()

                    # Add the failed item
                    result.failed.append((item, error))

                    # Finalize the result
                    result.total_time = time.time() - start_time
                    return result
            else:
                result.successful.append((item, item_result))

        # Calculate total processing time
        result.total_time = time.time() - start_time

        logger.info(
            "Batch processing completed: %s successful, %s failed in %.2fs",
            result.success_count,
            result.failure_count,
            result.total_time,
        )

        return result

    @staticmethod
    async def process_batch(
        items: Iterable[T],
        process_func: Callable[[list[T]], list[U]],
        batch_size: int = 50,
        delay_between_batches: float = 0.0,
        timeout: Optional[float] = None,
    ) -> BatchResult[T, U]:
        """
        Process items in batches.

        Args:
            items: Items to process
            process_func: Function that processes a batch of items
            batch_size: Number of items per batch
            delay_between_batches: Delay between batches in seconds
            timeout: Optional timeout per batch in seconds

        Returns:
            BatchResult containing successful and failed operations
        """
        result = BatchResult[T, U]()
        start_time = time.time()

        # Convert items to a list if it's not already
        items_list = list(items)
        total_items = len(items_list)

        if total_items == 0:
            logger.warning("No items to process")
            result.total_time = 0.0
            return result

        # Calculate number of batches
        batch_count = (total_items + batch_size - 1) // batch_size
        logger.info(
            "Processing %s items in %s batches of size %s",
            total_items,
            batch_count,
            batch_size,
        )

        # Process each batch
        for i in range(0, total_items, batch_size):
            batch = items_list[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                logger.info(
                    "Processing batch %s/%s with %s items",
                    batch_num,
                    batch_count,
                    len(batch),
                )

                # Handle both async and sync functions
                if asyncio.iscoroutinefunction(process_func):
                    if timeout:
                        batch_results = await asyncio.wait_for(
                            process_func(batch), timeout
                        )
                    else:
                        batch_results = await process_func(batch)
                else:
                    batch_copy = (
                        batch.copy()
                    )  # Create a copy to avoid lambda binding issues
                    if timeout:
                        batch_results = await asyncio.wait_for(
                            asyncio.to_thread(lambda b=batch_copy: process_func(b)),
                            timeout,
                        )
                    else:
                        batch_results = await asyncio.to_thread(
                            lambda b=batch_copy: process_func(b)
                        )

                # Ensure we got the same number of results as inputs
                if len(batch_results) != len(batch):
                    logger.warning(
                        "Batch %s returned %s results for %s inputs",
                        batch_num,
                        len(batch_results),
                        len(batch),
                    )

                # Record successful results
                for item, item_result in zip(batch, batch_results, strict=False):
                    result.successful.append((item, item_result))

            except asyncio.TimeoutError as timeout_err:
                logger.error("Timeout processing batch %s", batch_num)
                # Mark all items in the batch as failed
                for item in batch:
                    result.failed.append((item, timeout_err))
            except Exception as e:
                logger.error("Error processing batch %s: %s", batch_num, e)
                # Mark all items in the batch as failed
                for item in batch:
                    result.failed.append((item, e))

            # Apply delay between batches if specified
            if delay_between_batches > 0 and i + batch_size < total_items:
                await asyncio.sleep(delay_between_batches)

        # Calculate total processing time
        result.total_time = time.time() - start_time

        logger.info(
            "Batch processing completed: %s successful, %s failed in %.2fs",
            result.success_count,
            result.failure_count,
            result.total_time,
        )

        return result

    @staticmethod
    async def map_parallel(
        workflow: Workflow[Any, Any],
        items: Iterable[T],
        operation: Callable[[T], Any],
        max_concurrency: int = 10,
    ) -> list[Any]:
        """
        Map an operation across items in parallel using a workflow.

        This is a convenience method that uses the workflow's parallel execution
        capability to apply an operation to multiple items concurrently.

        Args:
            workflow: Workflow instance to use for parallel execution
            items: Items to process
            operation: Function to apply to each item
            max_concurrency: Maximum number of concurrent operations

        Returns:
            List of results in the same order as the input items
        """
        items_list = list(items)

        if not items_list:
            return []

        # Process in chunks to respect max_concurrency
        results = []
        for i in range(0, len(items_list), max_concurrency):
            chunk = items_list[i : i + max_concurrency]

            # Create coroutines for each item in the chunk
            coroutines = [
                (
                    operation(item)
                    if asyncio.iscoroutinefunction(operation)
                    else asyncio.to_thread(operation, item)
                )
                for item in chunk
            ]

            # Execute coroutines in parallel using the workflow
            chunk_results = await workflow.run_parallel(coroutines)
            results.extend(chunk_results)

        return results
