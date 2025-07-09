# API Models Reference

This page documents the API models used in FastADK's HTTP API.

## Request Models

::: fastadk.api.models.AgentRequest

::: fastadk.api.models.AgentStreamRequest

::: fastadk.api.models.ToolRequest

## Response Models

::: fastadk.api.models.AgentResponse

::: fastadk.api.models.ToolResponse

::: fastadk.api.models.ErrorResponse

## Tool Call Models

::: fastadk.api.models.ToolCall

::: fastadk.api.models.ToolCallResult

## Agent Information

::: fastadk.api.models.AgentInfo

## Health Check

::: fastadk.api.models.HealthCheck

## Examples

```python
# Example of using the API models directly
from fastadk.api.models import AgentRequest, AgentResponse

# Create a request
request = AgentRequest(
    message="What's the weather in San Francisco?",
    session_id="user-123",
    parameters={"temperature": 0.7}
)

# Create a response
response = AgentResponse(
    message="The weather in San Francisco is currently 65°F and partly cloudy.",
    session_id="user-123",
    tools_used=["get_weather"],
    conversation_id="conv-456"
)
```
