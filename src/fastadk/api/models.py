"""
API models for FastADK.

This module defines the request and response models for the FastADK API.
"""

from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Base model for agent API requests."""

    prompt: str = Field(..., description="The user's prompt or query for the agent")
    session_id: str | None = Field(
        None, description="Session identifier for stateful conversations"
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Additional options for the agent"
    )


class AgentStreamRequest(AgentRequest):
    """Request model for streaming agent responses."""

    stream: bool = Field(True, description="Whether to stream the response")


class ToolRequest(BaseModel):
    """Request model for direct tool execution."""

    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool"
    )
    session_id: str | None = Field(
        None, description="Session identifier for stateful executions"
    )


class AgentResponse(BaseModel):
    """Base model for agent API responses."""

    response: str = Field(..., description="The agent's response to the user")
    session_id: str = Field(
        ..., description="Session identifier for stateful conversations"
    )
    execution_time: float = Field(..., description="Execution time in seconds")
    tools_used: list[str] = Field(
        default_factory=list, description="Tools used in processing this request"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )


class ToolResponse(BaseModel):
    """Response model for tool execution."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: Any = Field(..., description="Result of the tool execution")
    execution_time: float = Field(..., description="Execution time in seconds")
    session_id: str | None = Field(
        None, description="Session identifier for stateful executions"
    )


class AgentInfo(BaseModel):
    """Information about an agent."""

    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    model: str = Field(..., description="The model used by the agent")
    provider: str = Field(..., description="The provider used by the agent")
    tools: list[dict[str, Any]] = Field(
        default_factory=list, description="Tools available to the agent"
    )


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status (ok, error)")
    version: str = Field(..., description="FastADK version")
    agents: int = Field(..., description="Number of registered agents")
    environment: str = Field(
        ..., description="Current environment (development, production)"
    )
    uptime: float = Field(..., description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    status_code: int = Field(..., description="HTTP status code")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional error details"
    )


class ToolCall(BaseModel):
    """Model representing a tool call."""

    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool call"
    )
    call_id: str = Field(..., description="Unique identifier for this tool call")


class ToolCallResult(BaseModel):
    """Model representing the result of a tool call."""

    tool_name: str = Field(..., description="Name of the tool that was called")
    result: Any = Field(..., description="Result of the tool call")
    call_id: str = Field(..., description="Identifier matching the original tool call")
    success: bool = Field(..., description="Whether the tool call was successful")
    error: str | None = Field(None, description="Error message if the call failed")
