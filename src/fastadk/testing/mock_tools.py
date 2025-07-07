"""
Mock LLM and Tool implementations for deterministic testing.

This module provides mock implementations of LLM providers and tools
that can be used for deterministic testing of FastADK agents.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, TypeVar

from fastadk.core.exceptions import ToolError

# Type variable for decorator function
F = TypeVar("F", bound=Callable[..., Any])


class MockLLM:
    """
    A deterministic mock LLM for testing agent behavior with predefined responses.

    This class provides a way to test agent behavior without calling real LLM APIs,
    with configurable response patterns and behavior.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "This is a mock response from the LLM.",
        throw_error: bool = False,
        error_rate: float = 0.0,
    ) -> None:
        """
        Initialize the MockLLM with predefined responses.

        Args:
            responses: A dictionary mapping prompt patterns (regex) to responses
            default_response: The default response when no pattern matches
            throw_error: Whether to throw errors for testing error handling
            error_rate: The rate at which to randomly throw errors (0.0-1.0)
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.throw_error = throw_error
        self.error_rate = error_rate
        self.history: List[Dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments passed to the LLM

        Returns:
            The generated response

        Raises:
            RuntimeError: If throw_error is True or based on error_rate
        """
        # Record the call
        self.history.append(
            {
                "prompt": prompt,
                "kwargs": kwargs,
                "timestamp": "2023-07-07T12:00:00Z",  # Example timestamp
            }
        )

        # Optionally throw errors for testing error handling
        if self.throw_error:
            raise RuntimeError("Simulated LLM error for testing")

        # Try to match against the response patterns
        for pattern, response in self.responses.items():
            if re.search(pattern, prompt, re.DOTALL):
                return response

        # Return the default response if no pattern matches
        return self.default_response

    async def stream(self, prompt: str, **kwargs: Any) -> "MockStream":
        """
        Stream a response for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments passed to the LLM

        Returns:
            A mock stream object
        """
        # Record the call
        self.history.append(
            {
                "prompt": prompt,
                "kwargs": kwargs,
                "streaming": True,
                "timestamp": "2023-07-07T12:00:00Z",  # Example timestamp
            }
        )

        response = ""
        # Try to match against the response patterns
        for pattern, resp in self.responses.items():
            if re.search(pattern, prompt, re.DOTALL):
                response = resp
                break

        if not response:
            response = self.default_response

        return MockStream(response)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of all calls made to this mock LLM."""
        return self.history

    def clear_history(self) -> None:
        """Clear the history of calls."""
        self.history = []

    def add_response(self, pattern: str, response: str) -> None:
        """
        Add a new response pattern.

        Args:
            pattern: The regex pattern to match against prompts
            response: The response to return when the pattern matches
        """
        self.responses[pattern] = response


class MockStream:
    """A mock stream for testing streaming responses."""

    def __init__(self, content: str, chunk_size: int = 10):
        """
        Initialize the mock stream.

        Args:
            content: The full content to stream
            chunk_size: The size of each chunk to yield
        """
        self.content = content
        self.chunk_size = chunk_size
        self._chunks = self._split_into_chunks()

    def _split_into_chunks(self) -> List[str]:
        """Split the content into chunks for streaming."""
        chunks = []
        for i in range(0, len(self.content), self.chunk_size):
            chunks.append(self.content[i : i + self.chunk_size])
        return chunks

    async def __aiter__(self):
        """Make this an async iterable."""
        return self

    async def __anext__(self):
        """Return the next chunk or raise StopAsyncIteration."""
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class MockTool:
    """
    A mock tool for testing agent behavior with predefined results.

    This class allows testing tool invocation without calling real external services.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Optional[Callable] = None,
        result_map: Optional[Dict[str, Any]] = None,
        default_result: Any = None,
        throw_error: bool = False,
    ) -> None:
        """
        Initialize the MockTool.

        Args:
            name: The name of the tool
            description: The description of what the tool does
            parameters: The parameters schema for the tool
            function: Optional function to dynamically compute results
            result_map: Mapping of parameter combinations to results
            default_result: Default result when no match in result_map
            throw_error: Whether to throw errors for testing error handling
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.result_map = result_map or {}
        self.default_result = default_result
        self.throw_error = throw_error
        self.invocations: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        """
        Call the tool with the given parameters.

        Args:
            **kwargs: The parameters for the tool

        Returns:
            The tool's result

        Raises:
            ToolError: If throw_error is True
        """
        # Record the invocation
        self.invocations.append(
            {
                "parameters": kwargs,
                "timestamp": "2023-07-07T12:00:00Z",  # Example timestamp
            }
        )

        # Optionally throw errors for testing error handling
        if self.throw_error:
            raise ToolError(
                message=f"Simulated error in tool '{self.name}'",
                error_code="MOCK_TOOL_ERROR",
                details={"parameters": kwargs},
            )

        # If a function is provided, use it to compute the result
        if self.function:
            return await self.function(**kwargs)

        # Try to match the parameters with the result map
        # Convert parameters to a JSON string for matching
        param_str = json.dumps(kwargs, sort_keys=True)

        for params_pattern, result in self.result_map.items():
            # Convert the pattern to a string for comparison if it's a dict
            if isinstance(params_pattern, dict):
                params_pattern = json.dumps(params_pattern, sort_keys=True)

            # Check if the parameters match the pattern
            if params_pattern in param_str or params_pattern == param_str:
                return result

        # Return the default result if no match
        return self.default_result

    def get_invocations(self) -> List[Dict[str, Any]]:
        """Get the history of all invocations of this tool."""
        return self.invocations

    def clear_invocations(self) -> None:
        """Clear the history of invocations."""
        self.invocations = []


def create_test_scenario(
    description: str,
    inputs: List[str],
    expected_outputs: List[str],
    expected_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a test scenario definition for use with the @test_scenario decorator.

    Args:
        description: Description of the test scenario
        inputs: List of user inputs to test
        expected_outputs: List of expected agent outputs
        expected_tools: Optional list of tools expected to be used

    Returns:
        A test scenario definition dictionary
    """
    return {
        "description": description,
        "inputs": inputs,
        "expected_outputs": expected_outputs,
        "expected_tools": expected_tools or [],
    }


def test_scenario(scenario: Dict[str, Any]) -> Callable[[F], F]:
    """
    Decorator for creating test scenarios with stubbed responses.

    Args:
        scenario: A test scenario definition created with create_test_scenario

    Returns:
        A decorator function that attaches metadata to the decorated function
    """

    # Using a shorter name for the inner function avoids the 'Unused variable' warning
    def inner(func: F) -> F:
        # Store the scenario metadata on the function
        func.test_scenario = scenario  # type: ignore
        return func

    return inner
