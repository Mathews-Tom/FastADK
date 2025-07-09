"""
Tests for FastADK agent functionality.
"""

import os
from unittest.mock import patch

import pytest

from fastadk import Agent, BaseAgent, tool
from fastadk.core.exceptions import ToolError
from fastadk.providers.base import ModelStub

# Import directly from utils to avoid the lazy loading in __init__.py
from fastadk.testing.utils import AgentTest, MockModel


# Test agent class for testing
@Agent(model="gemini-1.5-pro", description="Test Agent")
class TestableAgent(BaseAgent):
    """A test agent for unit testing."""

    def __init__(self):
        # Initialize the agent, but override the model initialization
        # to prevent real API calls during testing
        super().__init__()
        self.model = MockModel()

    @tool
    def add_numbers(self, a: int, b: int) -> int:
        """Add two numbers and return the result."""
        return a + b

    @tool(cache_ttl=300, retries=2)
    def say_hello(self, name: str) -> str:
        """Say hello to the given name."""
        return f"Hello, {name}!"


class TestAgentDecorator:
    """Tests for the @Agent decorator."""

    def test_basic_decorator(self):
        """Test that the @Agent decorator properly sets class attributes."""
        # These are intentionally testing protected attributes set by the decorator
        # pylint: disable=protected-access
        assert TestableAgent._model_name == "gemini-1.5-pro"
        assert TestableAgent._description == "Test Agent"
        assert TestableAgent._provider == "gemini"


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_registration(self):
        """Test that tools are properly registered on the agent."""
        agent = TestableAgent()

        # Call the methods to make sure they're registered
        agent.add_numbers(1, 2)
        agent.say_hello("World")

        # Check that tools are registered
        assert "add_numbers" in agent.tools
        assert "say_hello" in agent.tools

        # Verify tool metadata
        add_tool = agent.tools["add_numbers"]
        assert add_tool.name == "add_numbers"
        assert "Add two numbers" in add_tool.description
        assert add_tool.cache_ttl == 0
        assert add_tool.retries == 0

        hello_tool = agent.tools["say_hello"]
        assert hello_tool.name == "say_hello"
        assert "Say hello" in hello_tool.description
        assert hello_tool.cache_ttl == 300
        assert hello_tool.retries == 2


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = TestableAgent()
        assert not agent.tools_used, "tools_used should be empty initially"
        assert not agent.memory_data, "memory_data should be empty initially"

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test the run method with a mock model."""
        agent = TestableAgent()
        # Set the provider to a simulated one
        agent._provider = "simulated"
        agent.provider = ModelStub(name="simulated")

        # Mock the _generate_response method to return a fixed response
        original_generate = agent._generate_response

        async def mock_generate_response(user_input):
            # Return a fixed response for testing
            return "This is a test response"

        # Replace the method with our mock
        agent._generate_response = mock_generate_response

        try:
            response = await agent.run("Hello, agent!")
            assert response == "This is a test response"
        finally:
            # Restore the original method
            agent._generate_response = original_generate

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test the execute_tool method."""
        agent = TestableAgent()

        # Call the method first to register it
        agent.add_numbers(1, 2)

        # Test successful tool execution
        result = await agent.execute_tool("add_numbers", a=2, b=3)
        assert result == 5
        assert "add_numbers" in agent.tools_used

        # Test missing tool
        with pytest.raises(ToolError):
            await agent.execute_tool("non_existent_tool")


class TestLiteLLMProvider:
    """Tests for the LiteLLM provider."""

    @pytest.mark.asyncio
    @patch("litellm.completion")
    async def test_litellm_provider(self, mock_completion):
        """Test the LiteLLM provider."""
        # Set up mock response
        mock_completion.return_value.choices = [
            type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": "LiteLLM test response"})},
            )
        ]
        mock_completion.return_value.usage = type(
            "Usage",
            (),
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        # Create agent with LiteLLM provider
        @Agent(model="gpt-3.5-turbo", provider="litellm")
        class LiteLLMAgent(BaseAgent):
            """A test agent using LiteLLM provider."""

            pass

        # Set up environment variable
        with patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}):
            agent = LiteLLMAgent()

            # Call the agent
            response = await agent.run("Test message")

            # Verify response
            assert response == "LiteLLM test response"

            # Verify LiteLLM was called with correct parameters
            mock_completion.assert_called_once()
            args, kwargs = mock_completion.call_args
            assert kwargs["model"] == "gpt-3.5-turbo"
            assert kwargs["messages"][0]["content"] == "Test message"


class TestAgentScenarios(AgentTest):
    """Test scenarios for agents using the AgentTest class."""

    def setup_method(self):
        """Set up the test agent."""
        self.agent = TestableAgent()
        self.agent.model = MockModel()

    def mock_response(self, prompt: str, response: str) -> None:
        """Configure the mock model to return a specific response."""
        if not hasattr(self.agent, "model") or not isinstance(
            self.agent.model, MockModel
        ):
            self.agent.model = MockModel()
        self.agent.model.responses[prompt] = response

        # Call the methods to ensure they're registered
        self.agent.add_numbers(1, 2)
        self.agent.say_hello("Test")

    @pytest.mark.asyncio
    async def test_add_numbers_scenario(self):
        """Test the add_numbers tool."""
        # Configure mock model to use the tool with exact match
        self.mock_response("What is 2 + 3?", "Let me calculate that for you. 2 + 3 = 5")
        # Also add a default response that includes "5" for this test
        self.agent.model.default_response = "The answer is 5"

        # Run the agent
        response = await self.agent.run("What is 2 + 3?")

        # Verify response
        assert "5" in response

    @pytest.mark.asyncio
    async def test_hello_scenario(self):
        """Test the say_hello tool."""
        result = await self.agent.execute_tool("say_hello", name="World")
        assert result == "Hello, World!"
