# Testing Utilities

FastADK provides testing utilities to make it easier to test your agents and tools.

## Agent Testing

::: fastadk.testing.utils.AgentTest

::: fastadk.testing.utils.scenario

::: fastadk.testing.utils.load_test

## Mock Models

::: fastadk.testing.utils.MockModel

::: fastadk.testing.utils.MockResponse

## Example Usage

```python
import unittest
from fastadk.testing.utils import AgentTestCase, AgentTester
from fastadk.core import Agent, BaseAgent, tool

@Agent(model="gemini-2.0-pro")
class MathAgent(BaseAgent):
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

class TestMathAgent(AgentTestCase):
    def setUp(self):
        self.agent = MathAgent()
        self.tester = AgentTester(self.agent)
    
    def test_addition(self):
        # Test that the agent can add numbers
        response = self.tester.run("What is 2 + 3?")
        
        # Verify that the add tool was used
        self.tester.assert_tool_used("add")
        
        # Verify the response contains the expected result
        self.assertIn("5", response)
    
    def test_multiplication(self):
        # Test that the agent can multiply numbers
        response = self.tester.run("What is 4 times 7?")
        
        # Verify that the multiply tool was used
        self.tester.assert_tool_used("multiply")
        
        # Verify the response contains the expected result
        self.assertIn("28", response)

if __name__ == "__main__":
    unittest.main()
```

## Performance Testing

```python
from fastadk.testing.utils import PerformanceTester

def test_agent_performance():
    agent = MyAgent()
    tester = PerformanceTester(agent)
    
    # Test response time
    results = tester.measure_response_time(
        queries=["What's the weather?", "Tell me a joke"],
        iterations=10
    )
    
    print(f"Average response time: {results.average_time:.2f}s")
    print(f"p95 response time: {results.p95_time:.2f}s")
    
    # Test memory usage
    memory_results = tester.measure_memory_usage(
        query="What's the meaning of life?",
        iterations=5
    )
    
    print(f"Peak memory usage: {memory_results.peak_mb:.2f}MB")
```
