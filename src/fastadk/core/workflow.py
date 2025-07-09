"""
Workflow orchestration for FastADK.

This module provides a declarative way to define multi-agent workflows and
agent composition patterns.
"""

import asyncio
import enum
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from loguru import logger

from fastadk.core.exceptions import ValidationError, WorkflowError

# Type variable for workflow inputs and outputs
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class WorkflowStepStatus(str, enum.Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowResult(Generic[T]):
    """Result of a workflow execution."""

    output: T
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = field(default=0.0)
    step_results: dict[str, Any] = field(default_factory=dict)


class WorkflowStep(Generic[T, U], ABC):
    """
    Base class for all workflow steps.

    A workflow step is a single unit of work that takes an input and produces an output.
    Steps can be chained together to form a workflow.
    """

    def __init__(
        self,
        name: str | None = None,
        timeout: float | None = None,
        retry: int = 0,
    ):
        """
        Initialize a workflow step.

        Args:
            name: Optional name for this step
            timeout: Optional timeout in seconds
            retry: Number of retry attempts if the step fails
        """
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.timeout = timeout
        self.retry = retry
        self.status = WorkflowStepStatus.PENDING
        self.error: Exception | None = None
        self.result: Any = None
        self.execution_time: float = 0.0
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._dependencies: set[WorkflowStep] = set()
        self._dependents: set[WorkflowStep] = set()

    @abstractmethod
    async def execute(self, input_data: T) -> U:
        """
        Execute the workflow step.

        Args:
            input_data: Input data for the step

        Returns:
            Output data from the step
        """

    async def __call__(self, input_data: T) -> U:
        """
        Call the workflow step like a function.

        This implements the execution logic with timeout and retry handling.

        Args:
            input_data: Input data for the step

        Returns:
            Output data from the step

        Raises:
            WorkflowError: If the step fails after all retries
            asyncio.TimeoutError: If the step times out
        """
        self.status = WorkflowStepStatus.RUNNING
        self.start_time = time.time()
        remaining_retries = self.retry
        last_error = None

        while True:
            try:
                # Execute with timeout if specified
                if self.timeout:
                    result = await asyncio.wait_for(
                        self.execute(input_data), timeout=self.timeout
                    )
                else:
                    result = await self.execute(input_data)

                self.end_time = time.time()
                self.execution_time = self.end_time - self.start_time
                self.status = WorkflowStepStatus.COMPLETED
                self.result = result
                return result

            except Exception as e:
                last_error = e
                remaining_retries -= 1

                if remaining_retries < 0:
                    # No more retries, mark as failed
                    self.end_time = time.time()
                    self.execution_time = self.end_time - self.start_time
                    self.status = WorkflowStepStatus.FAILED
                    self.error = last_error

                    # Convert to WorkflowError
                    if isinstance(e, asyncio.TimeoutError):
                        error_message = (
                            f"Step '{self.name}' timed out after {self.timeout}s"
                        )
                        raise WorkflowError(
                            message=error_message,
                            error_code="WORKFLOW_STEP_TIMEOUT",
                            details={
                                "step_name": self.name,
                                "step_id": self.id,
                                "timeout": self.timeout,
                            },
                        ) from e
                    else:
                        error_message = f"Step '{self.name}' failed: {str(e)}"
                        raise WorkflowError(
                            message=error_message,
                            error_code="WORKFLOW_STEP_FAILED",
                            details={
                                "step_name": self.name,
                                "step_id": self.id,
                                "error": str(e),
                            },
                        ) from e

                # Log the error and retry
                logger.warning(
                    "Step '%s' failed: %s. " "Retrying (%d/%d)...",
                    self.name,
                    str(e),
                    self.retry - remaining_retries,
                    self.retry,
                )
                # Small delay before retry
                await asyncio.sleep(0.1 * (self.retry - remaining_retries))

    def depends_on(self, *steps: "WorkflowStep") -> "WorkflowStep":
        """
        Add dependencies to this step.

        Args:
            *steps: Steps that must complete before this step

        Returns:
            Self for method chaining
        """
        for step in steps:
            self._dependencies.add(step)
            step._dependents.add(self)
        return self

    def get_dependencies(self) -> set["WorkflowStep"]:
        """
        Get steps that this step depends on.

        Returns:
            Set of dependency steps
        """
        return self._dependencies.copy()

    def get_dependents(self) -> set["WorkflowStep"]:
        """
        Get steps that depend on this step.

        Returns:
            Set of dependent steps
        """
        return self._dependents.copy()

    def __rshift__(self, other: "WorkflowStep") -> "SequentialFlow":
        """
        Chain two steps together using the >> operator.

        This creates a sequential flow where the output of this step
        is passed as input to the next step.

        Args:
            other: The next step in the sequence

        Returns:
            A sequential flow containing both steps
        """
        return SequentialFlow(steps=[self, other])

    def __or__(self, other: "WorkflowStep") -> "ParallelFlow":
        """
        Combine two steps in parallel using the | operator.

        This creates a parallel flow where both steps are executed
        with the same input.

        Args:
            other: The step to execute in parallel

        Returns:
            A parallel flow containing both steps
        """
        return ParallelFlow(steps=[self, other])


class FunctionStep(WorkflowStep[T, U]):
    """
    A workflow step that executes a function.

    This step wraps a regular function or coroutine function and
    executes it as part of a workflow.
    """

    def __init__(
        self,
        func: Callable[[T], U],
        name: str | None = None,
        timeout: float | None = None,
        retry: int = 0,
    ):
        """
        Initialize a function step.

        Args:
            func: The function to execute
            name: Optional name for this step
            timeout: Optional timeout in seconds
            retry: Number of retry attempts if the step fails
        """
        super().__init__(name=name or func.__name__, timeout=timeout, retry=retry)
        self.func = func
        self._is_coroutine = asyncio.iscoroutinefunction(func)

    async def execute(self, input_data: T) -> U:
        """
        Execute the function.

        Args:
            input_data: Input data for the function

        Returns:
            Output data from the function
        """
        if self._is_coroutine:
            return await self.func(input_data)
        else:
            # Run sync function in a thread pool
            return await asyncio.to_thread(self.func, input_data)


class AgentStep(WorkflowStep[T, U]):
    """
    A workflow step that executes an agent.

    This step wraps an agent and calls it with the given input data.
    """

    def __init__(
        self,
        agent: Any,  # Agent type is dynamically checked
        method_name: str = "__call__",
        name: str | None = None,
        timeout: float | None = None,
        retry: int = 0,
    ):
        """
        Initialize an agent step.

        Args:
            agent: The agent to execute
            method_name: The method to call on the agent
            name: Optional name for this step
            timeout: Optional timeout in seconds
            retry: Number of retry attempts if the step fails
        """
        # Check if the object is an agent (has the required method)
        if not hasattr(agent, method_name):
            raise ValidationError(
                message=f"Agent does not have method '{method_name}'",
                error_code="INVALID_AGENT",
                details={"agent_type": type(agent).__name__},
            )

        super().__init__(
            name=name or f"{type(agent).__name__}.{method_name}",
            timeout=timeout,
            retry=retry,
        )
        self.agent = agent
        self.method_name = method_name
        self._method = getattr(agent, method_name)
        self._is_coroutine = asyncio.iscoroutinefunction(self._method)

    async def execute(self, input_data: T) -> U:
        """
        Execute the agent.

        Args:
            input_data: Input data for the agent

        Returns:
            Output data from the agent
        """
        if self._is_coroutine:
            return await self._method(input_data)
        else:
            # Run sync method in a thread pool
            return await asyncio.to_thread(self._method, input_data)


class TransformStep(WorkflowStep[T, U]):
    """
    A workflow step that transforms data.

    This step applies a transformation function to the input data
    without any external calls or side effects.
    """

    def __init__(
        self,
        transform_func: Callable[[T], U],
        name: str | None = None,
    ):
        """
        Initialize a transform step.

        Args:
            transform_func: The transformation function
            name: Optional name for this step
        """
        super().__init__(name=name or transform_func.__name__)
        self.transform_func = transform_func
        # Transform steps are fast and have no side effects, so no timeout or retry

    async def execute(self, input_data: T) -> U:
        """
        Execute the transformation.

        Args:
            input_data: Input data to transform

        Returns:
            Transformed output data
        """
        return self.transform_func(input_data)


class ConditionalStep(WorkflowStep[T, U]):
    """
    A workflow step that conditionally executes one of two paths.

    This step evaluates a condition function and then executes either
    the true_step or false_step based on the result.
    """

    def __init__(
        self,
        condition: Callable[[T], bool],
        true_step: WorkflowStep[T, U],
        false_step: WorkflowStep[T, U] | None = None,
        name: str | None = None,
    ):
        """
        Initialize a conditional step.

        Args:
            condition: Function that returns True or False
            true_step: Step to execute if condition is True
            false_step: Optional step to execute if condition is False
            name: Optional name for this step
        """
        super().__init__(name=name or "Conditional")
        self.condition = condition
        self.true_step = true_step
        self.false_step = false_step

    async def execute(self, input_data: T) -> U:
        """
        Evaluate condition and execute appropriate step.

        Args:
            input_data: Input data to evaluate and pass to selected step

        Returns:
            Output from the selected step, or input data if no step was selected
        """
        try:
            condition_result = self.condition(input_data)
        except Exception as e:
            raise WorkflowError(
                message=f"Error evaluating condition in step '{self.name}': {str(e)}",
                error_code="WORKFLOW_CONDITION_ERROR",
                details={"step_name": self.name, "step_id": self.id, "error": str(e)},
            ) from e

        if condition_result:
            return await self.true_step(input_data)
        elif self.false_step:
            return await self.false_step(input_data)
        else:
            # If no false step, just pass through the input data
            return cast(U, input_data)


class CompositeStep(WorkflowStep[T, U], ABC):
    """
    Base class for steps that contain other steps.

    This is an abstract base class for workflow step types that
    orchestrate multiple child steps, like sequential or parallel flows.
    """

    def __init__(
        self,
        steps: list[WorkflowStep],
        name: str | None = None,
        timeout: float | None = None,
    ):
        """
        Initialize a composite step.

        Args:
            steps: Child steps to execute
            name: Optional name for this step
            timeout: Optional timeout for the entire composite
        """
        super().__init__(name=name, timeout=timeout)
        self.steps = steps

    def add_step(self, step: WorkflowStep) -> "CompositeStep":
        """
        Add a step to the composite.

        Args:
            step: Step to add

        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self


class SequentialFlow(CompositeStep[T, Any]):
    """
    A workflow step that executes steps in sequence.

    This step executes a list of steps in order, passing the output
    of each step as input to the next step.
    """

    async def execute(self, input_data: T) -> Any:
        """
        Execute steps in sequence.

        Args:
            input_data: Input data for the first step

        Returns:
            Output from the last step
        """
        current_input = input_data
        last_output = None

        for step in self.steps:
            try:
                last_output = await step(current_input)
                current_input = last_output
            except Exception as e:
                # Wrap in WorkflowError if not already
                if not isinstance(e, WorkflowError):
                    raise WorkflowError(
                        message=f"Sequential flow step '{step.name}' failed: {str(e)}",
                        error_code="WORKFLOW_SEQUENTIAL_STEP_FAILED",
                        details={
                            "step_name": step.name,
                            "step_id": step.id,
                            "error": str(e),
                        },
                    ) from e
                raise

        return last_output

    def __rshift__(self, other: WorkflowStep) -> "SequentialFlow":
        """
        Add another step to the sequence using the >> operator.

        Args:
            other: Step to add to the sequence

        Returns:
            Self with the new step added
        """
        self.steps.append(other)
        return self


class ParallelFlow(CompositeStep[T, list[Any]]):
    """
    A workflow step that executes steps in parallel.

    This step executes a list of steps concurrently, passing the same
    input to each step and collecting all outputs.
    """

    async def execute(self, input_data: T) -> list[Any]:
        """
        Execute steps in parallel.

        Args:
            input_data: Input data for all steps

        Returns:
            List of outputs from all steps
        """
        # Create tasks for all steps
        tasks = [asyncio.create_task(step(input_data)) for step in self.steps]

        try:
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    step = self.steps[i]
                    if not isinstance(result, WorkflowError):
                        raise WorkflowError(
                            message=f"Parallel flow step '{step.name}' failed: {str(result)}",
                            error_code="WORKFLOW_PARALLEL_STEP_FAILED",
                            details={
                                "step_name": step.name,
                                "step_id": step.id,
                                "error": str(result),
                            },
                        ) from result
                    raise result

            return results
        except asyncio.CancelledError:
            # Cancel all tasks if the flow is cancelled
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    def __or__(self, other: WorkflowStep) -> "ParallelFlow":
        """
        Add another step to the parallel flow using the | operator.

        Args:
            other: Step to add to the parallel flow

        Returns:
            Self with the new step added
        """
        self.steps.append(other)
        return self


class MergeStep(WorkflowStep[list[Any], U]):
    """
    A workflow step that merges multiple inputs into a single output.

    This step is typically used after a ParallelFlow to combine the
    results of multiple parallel steps.
    """

    def __init__(
        self,
        merge_func: Callable[[list[Any]], U],
        name: str | None = None,
    ):
        """
        Initialize a merge step.

        Args:
            merge_func: Function that combines multiple inputs into one output
            name: Optional name for this step
        """
        super().__init__(name=name or merge_func.__name__)
        self.merge_func = merge_func

    async def execute(self, input_data: list[Any]) -> U:
        """
        Merge multiple inputs into a single output.

        Args:
            input_data: List of inputs to merge

        Returns:
            Merged output
        """
        return self.merge_func(input_data)


class Workflow(Generic[T, U]):
    """
    A complete workflow definition.

    This class represents a complete workflow with input and output types,
    and methods for executing and monitoring the workflow.
    """

    def __init__(
        self,
        root_step: WorkflowStep[T, U],
        name: str | None = None,
        description: str | None = None,
    ):
        """
        Initialize a workflow.

        Args:
            root_step: The starting step of the workflow
            name: Optional name for the workflow
            description: Optional description of the workflow
        """
        self.name = name or "Workflow"
        self.description = description
        self.root_step = root_step
        self.id = str(uuid.uuid4())
        self._execution_history: list[dict[str, Any]] = []

    async def execute(self, input_data: T) -> WorkflowResult[U]:
        """
        Execute the workflow.

        Args:
            input_data: Input data for the workflow

        Returns:
            Workflow result containing output and metadata
        """
        start_time = time.time()
        logger.info("Starting workflow '%s' with id %s", self.name, self.id)

        try:
            # Execute the root step
            output = await self.root_step(input_data)

            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Create result
            result = WorkflowResult(
                output=output,
                execution_time=execution_time,
                metadata={
                    "workflow_id": self.id,
                    "workflow_name": self.name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                # Include results from all steps
                step_results=self._collect_step_results(self.root_step),
            )

            # Log success
            logger.info(
                "Workflow '%s' completed successfully in %.2fs",
                self.name,
                execution_time,
            )

            # Record execution in history
            self._record_execution(True, execution_time, None)

            return result

        except Exception as e:
            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Log failure
            logger.error(
                "Workflow '%s' failed after %.2fs: %s",
                self.name,
                execution_time,
                str(e),
            )

            # Record execution in history
            self._record_execution(False, execution_time, str(e))

            # Re-raise the exception
            raise

    def _collect_step_results(self, step: WorkflowStep) -> dict[str, Any]:
        """
        Recursively collect results from all steps.

        Args:
            step: The step to collect results from

        Returns:
            Dictionary of step results
        """
        results = {
            step.name: {
                "id": step.id,
                "status": step.status,
                "execution_time": step.execution_time,
                "result": step.result,
                "error": str(step.error) if step.error else None,
            }
        }

        # If this is a composite step, collect results from child steps
        if isinstance(step, CompositeStep):
            for child_step in step.steps:
                results.update(self._collect_step_results(child_step))

        # If this is a conditional step, collect results from both branches
        if isinstance(step, ConditionalStep):
            results.update(self._collect_step_results(step.true_step))
            if step.false_step:
                results.update(self._collect_step_results(step.false_step))

        return results

    def _record_execution(
        self, success: bool, execution_time: float, error: str | None
    ) -> None:
        """
        Record workflow execution in history.

        Args:
            success: Whether the execution was successful
            execution_time: Time taken to execute
            error: Error message if execution failed
        """
        self._execution_history.append(
            {
                "timestamp": time.time(),
                "success": success,
                "execution_time": execution_time,
                "error": error,
            }
        )

        # Limit history size
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

    def get_execution_history(self) -> list[dict[str, Any]]:
        """
        Get workflow execution history.

        Returns:
            List of execution history entries
        """
        return self._execution_history.copy()

    def get_visualization(self) -> dict[str, Any]:
        """
        Get a representation of the workflow for visualization.

        Returns:
            Dictionary representation of the workflow structure
        """
        return self._visualize_step(self.root_step)

    def _visualize_step(self, step: "WorkflowStep") -> dict[str, Any]:
        """
        Recursively visualize a step and its children.

        Args:
            step: The step to visualize

        Returns:
            Dictionary representation of the step
        """
        step_data: dict[str, Any] = {
            "id": step.id,
            "name": step.name,
            "type": step.__class__.__name__,
        }

        # For composite steps, include children
        if isinstance(step, CompositeStep):
            child_visualizations = [self._visualize_step(child) for child in step.steps]
            step_data["children"] = child_visualizations

        # For conditional steps, include both branches
        elif isinstance(step, ConditionalStep):
            branches: dict[str, Any] = {
                "true": self._visualize_step(step.true_step),
            }
            if step.false_step:
                branches["false"] = self._visualize_step(step.false_step)

            step_data["branches"] = branches

        return step_data

    @classmethod
    def from_function(
        cls, func: Callable[[T], U], name: str | None = None
    ) -> "Workflow[T, U]":
        """
        Create a workflow from a single function.

        Args:
            func: The function to convert to a workflow
            name: Optional name for the workflow

        Returns:
            A workflow that executes the function
        """
        step = FunctionStep(func)
        return cls(root_step=step, name=name or func.__name__)

    @classmethod
    def sequence(cls, *steps: WorkflowStep, name: str | None = None) -> "Workflow":
        """
        Create a sequential workflow from steps.

        Args:
            *steps: Steps to execute in sequence
            name: Optional name for the workflow

        Returns:
            A workflow that executes the steps in sequence
        """
        if not steps:
            raise ValidationError(
                message="Cannot create a workflow with no steps",
                error_code="WORKFLOW_NO_STEPS",
            )

        # Create a sequential flow of all steps
        flow = SequentialFlow(steps=list(steps))
        return cls(root_step=flow, name=name)

    async def run_parallel(
        self, coroutines: list[Any], timeout: float | None = None
    ) -> list[Any]:
        """
        Execute a list of coroutines in parallel.

        This method is useful for running multiple operations concurrently
        without creating a full workflow step for each one.

        Args:
            coroutines: List of coroutine objects to execute
            timeout: Optional timeout in seconds for all coroutines

        Returns:
            List of results from all coroutines

        Raises:
            asyncio.TimeoutError: If the execution times out
            Exception: If any coroutine raises an exception
        """
        if not coroutines:
            return []

            logger.info(
                "Running %d coroutines in parallel for workflow '%s'",
                len(coroutines),
                self.name,
            )

        # Create tasks for all coroutines
        tasks = [asyncio.create_task(coro) for coro in coroutines]

        try:
            # Wait for all tasks with optional timeout
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks), timeout=timeout
                )
            else:
                results = await asyncio.gather(*tasks)

            return results
        except asyncio.TimeoutError:
            # Cancel all tasks if timeout
            for task in tasks:
                if not task.done():
                    task.cancel()

            logger.warning(
                "Parallel execution in workflow '%s' timed out after %ss",
                self.name,
                timeout,
            )
            raise
        except Exception as e:
            # Cancel remaining tasks if one fails
            for task in tasks:
                if not task.done():
                    task.cancel()

            logger.error("Error in parallel execution: %s", str(e))
            raise

    @classmethod
    def parallel(
        cls,
        *steps: WorkflowStep,
        merge_func: Callable[[list[Any]], Any] | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """
        Create a parallel workflow from steps.

        Args:
            *steps: Steps to execute in parallel
            merge_func: Optional function to merge results
            name: Optional name for the workflow

        Returns:
            A workflow that executes the steps in parallel
        """
        if not steps:
            raise ValidationError(
                message="Cannot create a workflow with no steps",
                error_code="WORKFLOW_NO_STEPS",
            )

        # Create a parallel flow of all steps
        parallel_flow = ParallelFlow(steps=list(steps))

        # If a merge function is provided, add a merge step
        if merge_func:
            merge_step = MergeStep(merge_func)
            # Create a sequential flow with the parallel flow followed by the merge step
            sequential_flow = SequentialFlow(steps=[parallel_flow, merge_step])
            return cls(root_step=sequential_flow, name=name)
        else:
            # Just use the parallel flow directly
            return cls(root_step=parallel_flow, name=name)


# Helper functions for creating workflow steps
def step(
    func: Callable | None = None,
    *,
    name: str | None = None,
    timeout: float | None = None,
    retry: int = 0,
) -> FunctionStep | Callable[[Callable], FunctionStep]:
    """
    Decorator to create a workflow step from a function.

    Can be used as @step or @step(name="Step Name", timeout=10)

    Args:
        func: The function to decorate
        name: Optional name for the step
        timeout: Optional timeout in seconds
        retry: Number of retry attempts if the step fails

    Returns:
        A workflow step or a decorator function
    """

    def decorator(f: Callable) -> FunctionStep:
        return FunctionStep(f, name=name, timeout=timeout, retry=retry)

    if func is None:
        return decorator
    else:
        return FunctionStep(func, name=name, timeout=timeout, retry=retry)


def transform(
    func: Callable | None = None,
    *,
    name: str | None = None,
) -> TransformStep | Callable[[Callable], TransformStep]:
    """
    Decorator to create a transform step from a function.

    Can be used as @transform or @transform(name="Transform Name")

    Args:
        func: The function to decorate
        name: Optional name for the step

    Returns:
        A transform step or a decorator function
    """

    def decorator(f: Callable) -> TransformStep:
        return TransformStep(f, name=name)

    if func is None:
        return decorator
    else:
        return TransformStep(func, name=name)


def conditional(
    condition: Callable[[Any], bool],
    true_step: WorkflowStep,
    false_step: WorkflowStep | None = None,
    name: str | None = None,
) -> ConditionalStep:
    """
    Create a conditional step.

    Args:
        condition: Function that returns True or False
        true_step: Step to execute if condition is True
        false_step: Optional step to execute if condition is False
        name: Optional name for the step

    Returns:
        A conditional step
    """
    return ConditionalStep(
        condition=condition,
        true_step=true_step,
        false_step=false_step,
        name=name,
    )


def merge(
    func: Callable | None = None,
    *,
    name: str | None = None,
) -> MergeStep | Callable[[Callable], MergeStep]:
    """
    Decorator to create a merge step from a function.

    Can be used as @merge or @merge(name="Merge Name")

    Args:
        func: The function to decorate
        name: Optional name for the step

    Returns:
        A merge step or a decorator function
    """

    def decorator(f: Callable) -> MergeStep:
        return MergeStep(f, name=name)

    if func is None:
        return decorator
    else:
        return MergeStep(func, name=name)
