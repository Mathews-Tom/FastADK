"""
Tests for workflow functionality.

This module tests the workflow orchestration capabilities.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from fastadk.core.exceptions import WorkflowError
from fastadk.core.workflow import (
    FunctionStep,
    MergeStep,
    ParallelFlow,
    SequentialFlow,
    TransformStep,
    Workflow,
    WorkflowStepStatus,
    conditional,
    merge,
    step,
    transform,
)


class TestWorkflowSteps:
    """Tests for workflow step classes."""

    @pytest.mark.asyncio
    async def test_function_step_sync(self):
        """Test function step with synchronous function."""

        # Create a function step
        def multiply(x):
            return x * 2

        func_step = FunctionStep(multiply)
        result = await func_step(5)
        assert result == 10
        assert func_step.status == WorkflowStepStatus.COMPLETED
        # Use >= 0 for platform compatibility (Windows may measure 0.0 for very fast operations)
        assert func_step.execution_time >= 0

    @pytest.mark.asyncio
    async def test_function_step_async(self):
        """Test function step with asynchronous function."""

        # Create an async function step
        async def add(x):
            await asyncio.sleep(0.01)  # Small delay for testing
            return x + 10

        func_step = FunctionStep(add)
        result = await func_step(5)
        assert result == 15
        assert func_step.status == WorkflowStepStatus.COMPLETED
        # Use >= 0 for platform compatibility (Windows may measure 0.0 for very fast operations)
        assert func_step.execution_time >= 0

    @pytest.mark.asyncio
    async def test_transform_step(self):
        """Test transform step."""

        # Create a transform step
        def format_name(data):
            return f"{data['first']} {data['last']}"

        transform_step = TransformStep(format_name)
        result = await transform_step({"first": "John", "last": "Doe"})
        assert result == "John Doe"
        assert transform_step.status == WorkflowStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_merge_step(self):
        """Test merge step."""

        # Create a merge step
        def sum_results(results):
            return sum(results)

        merge_step = MergeStep(sum_results)
        result = await merge_step([1, 2, 3, 4])
        assert result == 10
        assert merge_step.status == WorkflowStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_conditional_step_true_branch(self):
        """Test conditional step with true condition."""
        # Create steps for each branch
        true_step = FunctionStep(lambda x: x + 10)
        false_step = FunctionStep(lambda x: x - 10)

        # Create conditional step
        cond_step = conditional(
            condition=lambda x: x > 0,
            true_step=true_step,
            false_step=false_step,
        )

        # Execute with positive input (true branch)
        result = await cond_step(5)
        assert result == 15
        assert cond_step.status == WorkflowStepStatus.COMPLETED
        assert true_step.status == WorkflowStepStatus.COMPLETED
        assert false_step.status == WorkflowStepStatus.PENDING

    @pytest.mark.asyncio
    async def test_conditional_step_false_branch(self):
        """Test conditional step with false condition."""
        # Create steps for each branch
        true_step = FunctionStep(lambda x: x + 10)
        false_step = FunctionStep(lambda x: x - 10)

        # Create conditional step
        cond_step = conditional(
            condition=lambda x: x > 0,
            true_step=true_step,
            false_step=false_step,
        )

        # Execute with negative input (false branch)
        result = await cond_step(-5)
        assert result == -15
        assert cond_step.status == WorkflowStepStatus.COMPLETED
        assert true_step.status == WorkflowStepStatus.PENDING
        assert false_step.status == WorkflowStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_sequential_flow(self):
        """Test sequential flow."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)
        step3 = FunctionStep(lambda x: f"Result: {x}")

        # Create sequential flow
        flow = SequentialFlow(steps=[step1, step2, step3])
        result = await flow(5)
        assert result == "Result: 30"  # (5 + 10) * 2 = 30
        assert flow.status == WorkflowStepStatus.COMPLETED
        assert step1.status == WorkflowStepStatus.COMPLETED
        assert step2.status == WorkflowStepStatus.COMPLETED
        assert step3.status == WorkflowStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_flow(self):
        """Test parallel flow."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)
        step3 = FunctionStep(lambda x: x**2)

        # Create parallel flow
        flow = ParallelFlow(steps=[step1, step2, step3])
        results = await flow(5)
        assert results == [15, 10, 25]  # [5+10, 5*2, 5**2]
        assert flow.status == WorkflowStepStatus.COMPLETED
        assert step1.status == WorkflowStepStatus.COMPLETED
        assert step2.status == WorkflowStepStatus.COMPLETED
        assert step3.status == WorkflowStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_with_merge(self):
        """Test parallel flow with merge step."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)

        # Create merge step
        merge_step = MergeStep(lambda results: sum(results))

        # Create parallel flow and then sequence with merge
        parallel = ParallelFlow(steps=[step1, step2])
        flow = SequentialFlow(steps=[parallel, merge_step])

        result = await flow(5)
        assert result == 25  # (5+10) + (5*2) = 15 + 10 = 25

    @pytest.mark.asyncio
    async def test_step_retry(self):
        """Test step retry functionality."""
        # Create a mock function that fails twice then succeeds
        mock = AsyncMock(
            side_effect=[ValueError("Fail"), ValueError("Fail again"), "success"]
        )

        # Create function step with retry
        retry_step = FunctionStep(mock, retry=2)

        # Should succeed after retries
        result = await retry_step("input")
        assert result == "success"
        assert retry_step.status == WorkflowStepStatus.COMPLETED
        assert mock.call_count == 3  # Initial call + 2 retries

    @pytest.mark.asyncio
    async def test_step_retry_exhausted(self):
        """Test step retry exhausted."""
        # Create a mock function that always fails
        mock = AsyncMock(side_effect=ValueError("Always fails"))

        # Create function step with retry
        retry_step = FunctionStep(mock, retry=2)

        # Should fail after exhausting retries
        with pytest.raises(WorkflowError) as exc_info:
            await retry_step("input")

        assert retry_step.status == WorkflowStepStatus.FAILED
        assert mock.call_count == 3  # Initial call + 2 retries
        assert "failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_step_timeout(self):
        """Test step timeout functionality."""

        # Create a slow function
        async def slow_func(x):
            await asyncio.sleep(0.2)
            return x

        # Create function step with short timeout
        timeout_step = FunctionStep(slow_func, timeout=0.1)

        # Should timeout
        with pytest.raises(WorkflowError) as exc_info:
            await timeout_step("input")

        assert timeout_step.status == WorkflowStepStatus.FAILED
        assert "timed out" in str(exc_info.value)


class TestWorkflow:
    """Tests for the Workflow class."""

    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """Test a simple workflow with one step."""

        # Define a simple multiply function
        def multiply_by_two(x):
            return x * 2

        # Create a step
        multiply = FunctionStep(multiply_by_two)

        # Create workflow
        workflow = Workflow(root_step=multiply, name="Multiply Workflow")

        # Execute workflow
        result = await workflow.execute(5)

        assert result.output == 10
        # Use >= 0 for platform compatibility (Windows may measure 0.0 for very fast operations)
        assert result.execution_time >= 0
        assert "workflow_id" in result.metadata
        # Check step results - the name might be "<lambda>" instead of "Multiply"
        assert len(result.step_results) > 0

        # Check execution history
        history = workflow.get_execution_history()
        assert len(history) == 1
        assert history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_complex_workflow(self):
        """Test a more complex workflow with multiple steps."""
        # Create steps
        parse = FunctionStep(lambda x: {"value": int(x)})
        validate = FunctionStep(lambda x: x if x["value"] > 0 else {"value": 0})
        process = FunctionStep(lambda x: x["value"] * 2)
        format_result = FunctionStep(lambda x: f"Result: {x}")

        # Create sequential flow
        flow = SequentialFlow(steps=[parse, validate, process, format_result])

        # Create workflow
        workflow = Workflow(root_step=flow, name="Process Workflow")

        # Execute workflow
        result = await workflow.execute("5")

        assert result.output == "Result: 10"
        assert len(workflow.get_execution_history()) == 1

        # Check we have at least the sequential flow step result
        assert len(result.step_results) >= 1
        assert "SequentialFlow" in result.step_results

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling."""
        # Create a step that will fail
        failing_step = FunctionStep(lambda x: x / 0)

        # Create workflow
        workflow = Workflow(root_step=failing_step, name="Failing Workflow")

        # Execute workflow - should raise
        with pytest.raises(WorkflowError):
            await workflow.execute(10)

        # Check execution history
        history = workflow.get_execution_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert history[0]["error"] is not None

    @pytest.mark.asyncio
    async def test_workflow_from_function(self):
        """Test creating a workflow from a function."""

        # Create a workflow from a function
        async def process(x):
            await asyncio.sleep(0.01)
            return x * 2

        workflow = Workflow.from_function(process)

        # Execute workflow
        result = await workflow.execute(5)
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_workflow_sequence(self):
        """Test creating a sequential workflow."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)

        # Create workflow
        workflow = Workflow.sequence(step1, step2, name="Sequence Workflow")

        # Execute workflow
        result = await workflow.execute(5)
        assert result.output == 30  # (5 + 10) * 2 = 30

    @pytest.mark.asyncio
    async def test_workflow_parallel(self):
        """Test creating a parallel workflow."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)

        # Create workflow with merge function
        workflow = Workflow.parallel(
            step1,
            step2,
            merge_func=lambda results: sum(results),
            name="Parallel Workflow",
        )

        # Execute workflow
        result = await workflow.execute(5)
        assert result.output == 25  # (5 + 10) + (5 * 2) = 15 + 10 = 25

    @pytest.mark.asyncio
    async def test_workflow_visualization(self):
        """Test workflow visualization."""
        # Create a workflow with multiple steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)
        flow = SequentialFlow(steps=[step1, step2])

        workflow = Workflow(root_step=flow, name="Visualized Workflow")

        # Get visualization
        viz = workflow.get_visualization()

        assert viz["name"] == "SequentialFlow"
        assert "children" in viz
        assert len(viz["children"]) == 2
        assert viz["children"][0]["name"] == step1.name
        assert viz["children"][1]["name"] == step2.name


class TestWorkflowDecorators:
    """Tests for workflow decorator functions."""

    @pytest.mark.asyncio
    async def test_step_decorator(self):
        """Test @step decorator."""

        # Create a step with decorator
        @step
        def add_ten(x):
            return x + 10

        result = await add_ten(5)
        assert result == 15
        assert isinstance(add_ten, FunctionStep)

        # With parameters
        @step(name="Multiply", timeout=10, retry=2)
        def multiply(x):
            return x * 2

        result = await multiply(5)
        assert result == 10
        assert multiply.name == "Multiply"
        assert multiply.timeout == 10
        assert multiply.retry == 2

    @pytest.mark.asyncio
    async def test_transform_decorator(self):
        """Test @transform decorator."""

        # Create a transform with decorator
        @transform
        def format_name(data):
            return f"{data['first']} {data['last']}"

        result = await format_name({"first": "John", "last": "Doe"})
        assert result == "John Doe"
        assert isinstance(format_name, TransformStep)

        # With parameters
        @transform(name="Custom Transform")
        def upper_case(s):
            return s.upper()

        result = await upper_case("hello")
        assert result == "HELLO"
        assert upper_case.name == "Custom Transform"

    @pytest.mark.asyncio
    async def test_merge_decorator(self):
        """Test @merge decorator."""

        # Create a merge with decorator
        @merge
        def sum_all(values):
            return sum(values)

        result = await sum_all([1, 2, 3, 4])
        assert result == 10
        assert isinstance(sum_all, MergeStep)

        # With parameters
        @merge(name="Average")
        def average(values):
            return sum(values) / len(values)

        result = await average([10, 20, 30, 40])
        assert result == 25.0
        assert average.name == "Average"

    @pytest.mark.asyncio
    async def test_operator_overloading(self):
        """Test operator overloading for step composition."""
        # Create steps
        step1 = FunctionStep(lambda x: x + 10)
        step2 = FunctionStep(lambda x: x * 2)
        step3 = FunctionStep(lambda x: f"Result: {x}")

        # Chain with >> operator (sequential)
        sequential = step1 >> step2 >> step3
        assert isinstance(sequential, SequentialFlow)
        assert len(sequential.steps) == 3

        result = await sequential(5)
        assert result == "Result: 30"  # (5 + 10) * 2 = 30

        # Combine with | operator (parallel)
        parallel = step1 | step2
        assert isinstance(parallel, ParallelFlow)
        assert len(parallel.steps) == 2

        results = await parallel(5)
        assert results == [15, 10]  # [5+10, 5*2]
