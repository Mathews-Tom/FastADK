"""
Utilities for testing FastADK agents.

This module provides tools for mocking LLM APIs and testing agents.
"""

import asyncio
import inspect
import re
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from fastadk.core.agent import BaseAgent

# Type alias for clarity
Response = dict[str, Any]


class MockResponse:
    """Mock response from an LLM API."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text


class MockModel:
    """Mock model for testing agents without calling real LLM APIs."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "Mock response",
    ) -> None:
        """Initialize the mock model."""
        self.responses = responses or {}
        self.default_response = default_response
        self.invocations: list[dict[str, Any]] = []
        self.last_prompt: str | None = None

    def generate_content(self, prompt: str, **kwargs: Any) -> MockResponse:
        """Simulate generating content from a prompt."""
        self.invocations.append({"prompt": prompt, "params": kwargs})
        self.last_prompt = prompt

        # Try to find a matching response
        for pattern, response in self.responses.items():
            if re.search(pattern, prompt, re.DOTALL):
                return MockResponse(response)

        return MockResponse(self.default_response)


def scenario(
    name: str, description: str = "", **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for naming and documenting test scenarios."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Store metadata directly on the function
        func._scenario_name = name  # type: ignore
        func._scenario_description = description  # type: ignore
        # Add any other metadata
        for key, value in kwargs.items():
            setattr(func, f"_scenario_{key}", value)
        return func

    return decorator


def load_test(
    concurrent_users: int, duration: str, ramp_up: str = "0s", **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for creating load test scenarios."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Store load test metadata
        func._is_load_test = True  # type: ignore
        func._load_test_config = {  # type: ignore
            "concurrent_users": concurrent_users,
            "duration": duration,
            "ramp_up": ramp_up,
            **kwargs,
        }
        return func

    return decorator


class AgentTest:
    """Base class for testing FastADK agents."""

    # Reference to the agent class being tested
    agent_class: ClassVar[type[BaseAgent]]

    # Actual agent instance
    agent: BaseAgent

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Hook when a subclass is created."""
        super().__init_subclass__(**kwargs)

        # Automatically register pytest fixtures for scenarios
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if hasattr(method, "_scenario_name"):
                # Create a pytest fixture from this scenario
                cls._register_scenario_fixture(cls.__name__, name, method)

    @staticmethod
    def _register_scenario_fixture(
        class_name: str, scenario_name: str, scenario_method: Callable[..., Any]
    ) -> None:
        """Register a pytest fixture for a scenario method."""
        fixture_name = f"{class_name}_{scenario_name}"

        @pytest.fixture(name=fixture_name)
        async def _fixture() -> dict[str, Any]:
            # Create a new instance for each test
            instance = globals()[class_name]()

            if hasattr(instance, "setup"):
                await instance.setup()

            try:
                result = await scenario_method(instance)
                return {
                    "name": scenario_method._scenario_name,  # type: ignore
                    "description": getattr(
                        scenario_method, "_scenario_description", ""
                    ),
                    "result": result,
                    "success": True,
                }
            except Exception as e:
                return {
                    "name": scenario_method._scenario_name,  # type: ignore
                    "description": getattr(
                        scenario_method, "_scenario_description", ""
                    ),
                    "error": str(e),
                    "success": False,
                }
            finally:
                if hasattr(instance, "teardown"):
                    await instance.teardown()

        # The fixture is automatically registered with pytest due to the decorator

    async def setup(self) -> None:
        """Set up the test environment before each test."""

    async def teardown(self) -> None:
        """Clean up the test environment after each test."""

    def assert_tools_used(self, tool_names: list[str]) -> None:
        """Verify that specific tools were used during the test."""
        if not hasattr(self.agent, "tools_used"):
            pytest.fail("Agent does not track tools used")

        for name in tool_names:
            assert name in self.agent.tools_used, f"Tool '{name}' was not used"

    def configure_mock_model(self, prompt: str, response: str) -> None:
        """Configure the mock model to return a specific response for a prompt."""
        if not hasattr(self.agent, "model") or not isinstance(
            self.agent.model, MockModel
        ):
            self.agent.model = MockModel()  # type: ignore

        # Add the response to the model's response dictionary
        self.agent.model.responses[prompt] = response  # type: ignore[attr-defined]

    @classmethod
    async def run_load_test(cls, scenario_name: str, **kwargs: Any) -> dict[str, Any]:
        """Run a load test scenario with multiple concurrent users."""
        # Get the scenario method
        scenario_method = None
        for _name, method in inspect.getmembers(cls, inspect.isfunction):
            if (
                hasattr(method, "_scenario_name")
                and method._scenario_name == scenario_name  # type: ignore
                and hasattr(method, "_is_load_test")
            ):
                scenario_method = method
                break

        if not scenario_method:
            raise ValueError(f"No load test scenario found with name '{scenario_name}'")

        # Get load test config
        config = {**scenario_method._load_test_config}  # type: ignore
        config.update(kwargs)

        concurrent_users = config.get("concurrent_users", 10)

        # Create tasks for each user
        tasks = []
        for _ in range(concurrent_users):
            instance = cls()
            await instance.setup()
            tasks.append(scenario_method(instance))

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        failure_count = len(results) - success_count

        return {
            "scenario": scenario_name,
            "config": config,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(results) if results else 0,
            "results": results,
        }
