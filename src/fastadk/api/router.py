"""
FastAPI router for FastADK.

This module provides the FastAPI router for serving FastADK agents via HTTP.
"""

import logging
import time
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse

from fastadk import __version__
from fastadk.core.agent import BaseAgent
from fastadk.core.config import get_settings
from fastadk.core.exceptions import AgentError, ToolError

from .models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    HealthCheck,
    ToolRequest,
    ToolResponse,
)

# Set up logging
logger = logging.getLogger("fastadk.api")


class AgentRegistry:
    """Registry for FastADK agents."""

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agents: dict[str, type[BaseAgent]] = {}
        self._instances: dict[str, dict[str, BaseAgent]] = {}
        self._start_time = time.time()

    def register(self, agent_class: type[BaseAgent]) -> None:
        """
        Register an agent class with the registry.

        Args:
            agent_class: The agent class to register
        """
        name = agent_class.__name__
        self._agents[name] = agent_class
        self._instances[name] = {}
        logger.info(f"Registered agent: {name}")

    def get_agent_class(self, name: str) -> type[BaseAgent]:
        """
        Get an agent class by name.

        Args:
            name: The name of the agent class

        Returns:
            The agent class

        Raises:
            HTTPException: If the agent is not found
        """
        if name not in self._agents:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return self._agents[name]

    def get_agent_instance(self, name: str, session_id: str) -> BaseAgent:
        """
        Get or create an agent instance for a session.

        Args:
            name: The name of the agent class
            session_id: The session ID

        Returns:
            The agent instance
        """
        agent_class = self.get_agent_class(name)

        # Create a new session-specific agent instance if needed
        if session_id not in self._instances[name]:
            self._instances[name][session_id] = agent_class()
            self._instances[name][session_id].session_id = session_id
            logger.debug(f"Created new instance of {name} for session {session_id}")

        return self._instances[name][session_id]

    def list_agents(self) -> list[AgentInfo]:
        """
        List all registered agents.

        Returns:
            A list of agent information
        """
        result = []
        for name, agent_class in self._agents.items():
            tools = []
            # Create a temporary instance to inspect tools
            temp_instance = agent_class()
            for tool_name, tool in temp_instance.tools.items():
                tools.append(
                    {
                        "name": tool_name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )

            result.append(
                AgentInfo(
                    name=name,
                    description=agent_class._description,
                    model=agent_class._model_name,
                    provider=agent_class._provider,
                    tools=tools,
                )
            )
        return result

    def get_health_check(self) -> HealthCheck:
        """
        Get a health check response.

        Returns:
            Health check information
        """
        return HealthCheck(
            status="ok",
            version=__version__,
            agents=len(self._agents),
            environment=get_settings().environment,
            uptime=time.time() - self._start_time,
        )

    def clear_session(self, name: str, session_id: str) -> None:
        """
        Clear a session for an agent.

        Args:
            name: The name of the agent class
            session_id: The session ID
        """
        if name in self._instances and session_id in self._instances[name]:
            del self._instances[name][session_id]
            logger.debug(f"Cleared session {session_id} for agent {name}")


# Create a global registry
registry = AgentRegistry()


def create_api_router() -> APIRouter:
    """
    Create the FastAPI router for FastADK.

    Returns:
        The FastAPI router
    """
    router = APIRouter(tags=["FastADK Agents"])

    @router.get("/", response_model=HealthCheck)
    async def health_check() -> HealthCheck:
        """
        Health check endpoint.

        Returns:
            Health check information
        """
        return registry.get_health_check()

    @router.get("/agents", response_model=list[AgentInfo])
    async def list_agents() -> list[AgentInfo]:
        """
        List all registered agents.

        Returns:
            A list of agent information
        """
        return registry.list_agents()

    @router.get("/agents/{agent_name}", response_model=AgentInfo)
    async def get_agent_info(
        agent_name: str = Path(..., description="Name of the agent")
    ) -> AgentInfo:
        """
        Get information about a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent information
        """
        agent_class = registry.get_agent_class(agent_name)
        tools = []

        # Create a temporary instance to inspect tools
        temp_instance = agent_class()
        for tool_name, tool in temp_instance.tools.items():
            tools.append(
                {
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )

        return AgentInfo(
            name=agent_name,
            description=agent_class._description,
            model=agent_class._model_name,
            provider=agent_class._provider,
            tools=tools,
        )

    @router.post("/agents/{agent_name}", response_model=AgentResponse)
    async def run_agent(
        request: AgentRequest,
        agent_name: str = Path(..., description="Name of the agent"),
    ) -> AgentResponse:
        """
        Run an agent with the given input.

        Args:
            request: The agent request
            agent_name: Name of the agent

        Returns:
            The agent's response
        """
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            agent = registry.get_agent_instance(agent_name, session_id)
            agent.on_start()

            response = await agent.run(request.prompt)

            execution_time = time.time() - start_time
            logger.info(f"Agent {agent_name} completed in {execution_time:.2f}s")

            # Call the on_finish hook
            agent.on_finish(response)

            return AgentResponse(
                response=response,
                session_id=session_id,
                execution_time=execution_time,
                tools_used=agent.tools_used,
                meta={},
            )

        except AgentError as e:
            logger.error(f"Agent error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @router.post("/agents/{agent_name}/tools", response_model=ToolResponse)
    async def execute_tool(
        request: ToolRequest,
        agent_name: str = Path(..., description="Name of the agent"),
    ) -> ToolResponse:
        """
        Execute a specific tool of an agent.

        Args:
            request: The tool request
            agent_name: Name of the agent

        Returns:
            The tool's response
        """
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            agent = registry.get_agent_instance(agent_name, session_id)

            result = await agent.execute_tool(request.tool_name, **request.parameters)

            execution_time = time.time() - start_time
            logger.info(f"Tool {request.tool_name} completed in {execution_time:.2f}s")

            return ToolResponse(
                tool_name=request.tool_name,
                result=result,
                execution_time=execution_time,
                session_id=session_id,
            )

        except ToolError as e:
            logger.error(f"Tool error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    @router.get("/stream/agents/{agent_name}")
    async def stream_agent(
        prompt: str = Query(..., description="The user's prompt or query"),
        agent_name: str = Path(..., description="Name of the agent"),
        session_id: str | None = Query(None, description="Session identifier"),
    ) -> StreamingResponse:
        """
        Stream an agent's response for the given input.

        Args:
            prompt: The user's prompt
            agent_name: Name of the agent
            session_id: Session identifier

        Returns:
            A streaming response with the agent's output
        """

        # This is a placeholder for streaming functionality
        # Actual implementation will be added in Phase 3
        # For now, return a simple message
        async def fake_stream():
            yield f"Streaming not yet implemented for agent {agent_name}\n"
            yield "This feature will be available in Phase 3\n"

        return StreamingResponse(fake_stream(), media_type="text/event-stream")

    @router.delete("/agents/{agent_name}/sessions/{session_id}")
    async def clear_agent_session(
        agent_name: str = Path(..., description="Name of the agent"),
        session_id: str = Path(..., description="Session identifier"),
    ) -> JSONResponse:
        """
        Clear a session for an agent.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier

        Returns:
            A confirmation message
        """
        try:
            registry.clear_session(agent_name, session_id)
            return JSONResponse(
                {"status": "success", "message": f"Session {session_id} cleared"}
            )
        except Exception as e:
            logger.exception(f"Error clearing session: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error clearing session: {str(e)}"
            )

    return router


def create_app() -> FastAPI:
    """
    Create the FastAPI application for FastADK.

    Returns:
        The FastAPI application
    """
    app = FastAPI(
        title="FastADK API",
        description="API for FastADK Agents",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add the FastADK router
    app.include_router(create_api_router())

    @app.on_event("startup")
    async def startup_event():
        logger.info("FastADK API starting up")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("FastADK API shutting down")

    return app
