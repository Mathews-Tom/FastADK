"""
Streaming API integration for FastADK.

This module provides streaming capabilities for FastADK agents via FastAPI,
supporting both Server-Sent Events (SSE) and WebSockets.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Try to import SSE
try:
    from sse_starlette.sse import EventSourceResponse

    _has_sse = True
except ImportError:
    _has_sse = False

    # Simple SSE response implementation
    class EventSourceResponse(StreamingResponse):
        """Simple Server-Sent Events response."""

        def __init__(self, content, **kwargs):
            super().__init__(content, media_type="text/event-stream", **kwargs)


from fastadk.api.models import AgentRequest
from fastadk.core.agent import get_registered_agent
from fastadk.core.exceptions import FastADKError


class StreamType(str, Enum):
    """Types of supported streaming responses."""

    SSE = "sse"
    WEBSOCKET = "websocket"


class StreamEvent(BaseModel):
    """A single event in a stream."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event: str = "message"
    data: Any
    retry: int | None = None
    timestamp: float = Field(default_factory=time.time)


class StreamingManager:
    """
    Manager for streaming connections.

    This class handles all active streaming connections and provides
    methods for sending events to them.
    """

    def __init__(self):
        """Initialize the streaming manager."""
        self._active_connections: dict[str, WebSocket] = {}
        self._event_channels: dict[str, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self._active_connections[client_id] = websocket
        self._event_channels[client_id] = asyncio.Queue()
        logger.info("Client %s connected via WebSocket", client_id)

        try:
            # Send welcome message
            await websocket.send_json(
                {
                    "event": "connected",
                    "data": {
                        "client_id": client_id,
                        "message": "Connected to FastADK streaming API",
                    },
                }
            )

            # Process events from the queue
            while True:
                event = await self._event_channels[client_id].get()
                await websocket.send_json(
                    {
                        "id": event.id,
                        "event": event.event,
                        "data": event.data,
                        "timestamp": event.timestamp,
                    }
                )
        except WebSocketDisconnect:
            logger.info("Client %s disconnected", client_id)
            self.disconnect(client_id)
        except Exception as e:
            logger.error("Error in WebSocket connection: %s", str(e))
            self.disconnect(client_id)

    def disconnect(self, client_id: str) -> None:
        """
        Disconnect a client.

        Args:
            client_id: The client ID to disconnect
        """
        self._active_connections.pop(client_id, None)
        self._event_channels.pop(client_id, None)
        logger.info("Client %s removed from active connections", client_id)

    async def send_event(self, client_id: str, event: StreamEvent) -> bool:
        """
        Send an event to a specific client.

        Args:
            client_id: The client to send to
            event: The event to send

        Returns:
            True if sent successfully, False if client not found
        """
        if client_id in self._event_channels:
            await self._event_channels[client_id].put(event)
            return True
        return False

    async def broadcast_event(
        self, event: StreamEvent, exclude: list[str] | None = None
    ) -> int:
        """
        Broadcast an event to all connected clients.

        Args:
            event: The event to broadcast
            exclude: Optional list of client IDs to exclude

        Returns:
            Number of clients the event was sent to
        """
        exclude = exclude or []
        count = 0

        for client_id in list(self._event_channels.keys()):
            if client_id not in exclude:
                if await self.send_event(client_id, event):
                    count += 1

        return count

    def get_active_clients(self) -> list[str]:
        """
        Get a list of active client IDs.

        Returns:
            List of client IDs
        """
        return list(self._active_connections.keys())

    def get_client_count(self) -> int:
        """
        Get the count of active clients.

        Returns:
            Number of active clients
        """
        return len(self._active_connections)


# Global streaming manager instance
streaming_manager = StreamingManager()


async def generate_sse_events(
    agent_name: str, request: AgentRequest, stream_id: str
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events (SSE) for an agent request.

    Args:
        agent_name: Name of the agent to execute
        request: The agent request data
        stream_id: Unique identifier for this stream

    Yields:
        SSE formatted event strings
    """
    try:
        # Get the agent
        agent_class = get_registered_agent(agent_name)
        if not agent_class:
            yield f"data: {json.dumps({'error': f'Agent {agent_name} not found'})}\n\n"
            return

        # Create and initialize the agent
        agent = agent_class()

        # Start streaming response
        yield f"data: {json.dumps({'event': 'start', 'stream_id': stream_id})}\n\n"

        # Stream the thinking process if supported by the agent
        if hasattr(agent, "stream_thinking"):
            # Create a queue for the thinking events
            thinking_queue: asyncio.Queue = asyncio.Queue()

            # Start the agent thinking in a background task
            async def run_agent():
                try:
                    response = await agent(
                        request.input,
                        stream_thinking=thinking_queue.put,
                        **request.parameters,
                    )
                    # Put the final response in the queue
                    await thinking_queue.put(
                        {
                            "event": "complete",
                            "data": (
                                response.dict()
                                if hasattr(response, "dict")
                                else response
                            ),
                        }
                    )
                except Exception as e:
                    # Put the error in the queue
                    error_message = str(e)
                    error_type = type(e).__name__
                    if isinstance(e, FastADKError):
                        error_code = getattr(e, "error_code", "UNKNOWN_ERROR")
                        details = getattr(e, "details", {})
                    else:
                        error_code = "UNHANDLED_ERROR"
                        details = {}

                    await thinking_queue.put(
                        {
                            "event": "error",
                            "data": {
                                "message": error_message,
                                "type": error_type,
                                "error_code": error_code,
                                "details": details,
                            },
                        }
                    )

            # Start the agent execution in the background
            asyncio.create_task(run_agent())

            # Stream thinking events as they come in
            while True:
                event = await thinking_queue.get()
                event_type = event.get("event", "thinking")

                # Format as SSE
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event.get('data', ''))}\n\n"

                # If this is a completion or error event, we're done
                if event_type in ["complete", "error"]:
                    break
        else:
            # For agents that don't support streaming, just get the final result
            try:
                response = await agent(request.input, **request.parameters)
                yield "event: complete\n"
                yield f"data: {json.dumps(response.dict() if hasattr(response, 'dict') else response)}\n\n"
            except Exception as e:
                # Handle errors
                error_message = str(e)
                error_type = type(e).__name__
                if isinstance(e, FastADKError):
                    error_code = getattr(e, "error_code", "UNKNOWN_ERROR")
                    details = getattr(e, "details", {})
                else:
                    error_code = "UNHANDLED_ERROR"
                    details = {}

                yield "event: error\n"
                yield f"data: {
                    json.dumps(
                        {
                            'message': error_message,
                            'type': error_type,
                            'error_code': error_code,
                            'details': details,
                        }
                    )
                }\n\n"

    except Exception as e:
        # Catch any unexpected errors
        logger.exception("Error in SSE stream for agent %s: %s", agent_name, str(e))
        yield "event: error\n"
        yield f"data: {json.dumps({'message': f'Internal server error: {str(e)}'})}\n\n"


def create_streaming_router() -> APIRouter:
    """
    Create a FastAPI router for streaming endpoints.

    Returns:
        Router with streaming endpoints configured
    """
    if not _has_sse:
        logger.warning(
            "SSE functionality limited: sse-starlette not installed. Install with: uv add sse-starlette"
        )

    router = APIRouter()

    @router.get("/agents/{agent_name}/stream", response_class=StreamingResponse)
    async def stream_agent_sse(agent_name: str, input: str):
        """
        Stream agent responses using Server-Sent Events (SSE).

        Args:
            agent_name: Name of the registered agent
            input: Input text for the agent

        Returns:
            SSE streaming response
        """
        stream_id = str(uuid.uuid4())
        request = AgentRequest(input=input)

        return EventSourceResponse(
            generate_sse_events(agent_name, request, stream_id),
            media_type="text/event-stream",
        )

    @router.post("/agents/{agent_name}/stream", response_class=StreamingResponse)
    async def stream_agent_sse_post(agent_name: str, request: AgentRequest):
        """
        Stream agent responses using Server-Sent Events (SSE) with POST request.

        Args:
            agent_name: Name of the registered agent
            request: The agent request data

        Returns:
            SSE streaming response
        """
        stream_id = str(uuid.uuid4())

        return EventSourceResponse(
            generate_sse_events(agent_name, request, stream_id),
            media_type="text/event-stream",
        )

    @router.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """
        WebSocket endpoint for bidirectional communication.

        Args:
            websocket: The WebSocket connection
            client_id: Unique client identifier
        """
        await streaming_manager.connect(websocket, client_id)

    @router.post("/agents/{agent_name}/websocket/{client_id}")
    async def trigger_agent_websocket(
        agent_name: str, client_id: str, request: AgentRequest
    ):
        """
        Trigger an agent execution and send results via WebSocket.

        Args:
            agent_name: Name of the registered agent
            client_id: Client ID to send results to
            request: The agent request data

        Returns:
            Acknowledgement of the request
        """
        # Check if client is connected
        if client_id not in streaming_manager.get_active_clients():
            raise HTTPException(
                status_code=404, detail=f"Client {client_id} not connected"
            )

        # Get the agent
        agent_class = get_registered_agent(agent_name)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        # Start execution in background task
        async def execute_agent():
            try:
                # Create and initialize the agent
                agent = agent_class()

                # Send start event
                await streaming_manager.send_event(
                    client_id,
                    StreamEvent(
                        event="start",
                        data={"agent": agent_name, "request_id": str(uuid.uuid4())},
                    ),
                )

                # Stream thinking if supported
                if hasattr(agent, "stream_thinking"):

                    async def send_thinking(thinking_data):
                        await streaming_manager.send_event(
                            client_id,
                            StreamEvent(event="thinking", data=thinking_data),
                        )

                    # Execute agent with streaming
                    response = await agent(
                        request.input,
                        stream_thinking=send_thinking,
                        **request.parameters,
                    )
                else:
                    # Execute agent without streaming
                    response = await agent(request.input, **request.parameters)

                # Send completion event
                await streaming_manager.send_event(
                    client_id,
                    StreamEvent(
                        event="complete",
                        data=response.dict() if hasattr(response, "dict") else response,
                    ),
                )

            except Exception as e:
                # Handle errors
                error_message = str(e)
                error_type = type(e).__name__
                if isinstance(e, FastADKError):
                    error_code = getattr(e, "error_code", "UNKNOWN_ERROR")
                    details = getattr(e, "details", {})
                else:
                    error_code = "UNHANDLED_ERROR"
                    details = {}

                await streaming_manager.send_event(
                    client_id,
                    StreamEvent(
                        event="error",
                        data={
                            "message": error_message,
                            "type": error_type,
                            "error_code": error_code,
                            "details": details,
                        },
                    ),
                )

        # Start execution in background
        asyncio.create_task(execute_agent())

        # Return acknowledgement
        return {
            "status": "processing",
            "message": f"Agent {agent_name} execution started, results will be sent via WebSocket",
            "client_id": client_id,
        }

    return router
