"""
FastAPI router for FastADK.

This module provides the FastAPI router for serving FastADK agents via HTTP.
"""

import logging
import time
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

try:
    import sse_starlette.sse  # noqa: F401

    _has_sse = True
except ImportError:
    _has_sse = False

from fastadk import __version__
from fastadk.core.agent import BaseAgent
from fastadk.core.config import get_settings
from fastadk.core.exceptions import (
    AgentError,
    AuthenticationError,
    FastADKError,
    NotFoundError,
    OperationTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    ToolError,
    ValidationError,
)
from fastadk.observability.metrics import metrics

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
        logger.info("Registered agent: %s", name)

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
            logger.debug("Created new instance of %s for session %s", name, session_id)

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
            logger.debug("Cleared session %s for agent %s", session_id, name)


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
        agent_name: str = Path(..., description="Name of the agent"),
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
            logger.info("Agent %s completed in %.2fs", agent_name, execution_time)

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
            logger.error("Agent error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            error_msg = str(e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {error_msg}"
            ) from e

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
            logger.info("Tool %s completed in %.2fs", request.tool_name, execution_time)

            return ToolResponse(
                tool_name=request.tool_name,
                result=result,
                execution_time=execution_time,
                session_id=session_id,
            )

        except ToolError as e:
            logger.error("Tool error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            ) from e

    # Remove placeholder streaming endpoint as it's now implemented in streaming.py

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
            logger.exception("Error clearing session: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Error clearing session: {str(e)}"
            ) from e

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

    # Add the FastADK routers
    api_router = create_api_router()
    app.include_router(api_router)

    # Add streaming router if SSE is available
    if _has_sse:
        try:
            from .streaming import create_streaming_router

            streaming_router = create_streaming_router()
            app.include_router(streaming_router, tags=["FastADK Streaming"])
            logger.info("Streaming API endpoints enabled")
        except ImportError as e:
            logger.warning("Streaming API endpoints disabled: %s", str(e))

    # Add exception handlers
    @app.exception_handler(FastADKError)
    async def fastadk_exception_handler(_: Request, exc: FastADKError) -> JSONResponse:
        """Handle FastADK exceptions and convert to appropriate HTTP responses."""
        status_code = 500

        # Map exception types to status codes
        if isinstance(exc, ValidationError) or isinstance(exc, RequestValidationError):
            status_code = 400
        elif isinstance(exc, AuthenticationError):
            status_code = 401
        elif isinstance(exc, RateLimitError):
            status_code = 429
        elif isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, OperationTimeoutError):
            status_code = 408
        elif isinstance(exc, ServiceUnavailableError):
            status_code = 503

        # Prepare the response
        return JSONResponse(
            status_code=status_code,
            content={
                "error": True,
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "type": exc.__class__.__name__,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        # Convert validation error to FastADK format
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "message": "Request validation failed",
                "error_code": "REQUEST_VALIDATION_ERROR",
                "details": {"errors": exc.errors(), "body": exc.body},
                "type": "ValidationError",
            },
        )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        """
        Prometheus metrics endpoint.

        Returns:
            Metrics in Prometheus format
        """
        return Response(
            content=metrics.generate_latest(), media_type=metrics.content_type()
        )

    @app.get("/config/reload", response_model=dict)
    async def reload_config() -> dict:
        """
        Reload the configuration from disk.

        This endpoint is useful when you have modified the configuration file
        and want to apply changes without restarting the application.

        Returns:
            A confirmation message
        """
        try:
            # Import here to avoid circular imports
            from fastadk.core.config import reload_settings

            # Reload the settings
            old_settings = get_settings()
            new_settings = reload_settings()

            # Return info about what changed
            return {
                "status": "success",
                "message": "Configuration reloaded successfully",
                "old_environment": old_settings.environment,
                "new_environment": new_settings.environment,
                "config_path": new_settings.config_path or "No config file found",
            }
        except Exception as e:
            logger.exception("Error reloading configuration: %s", e)
            return {
                "status": "error",
                "message": f"Error reloading configuration: {str(e)}",
            }

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("FastADK API starting up")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("FastADK API shutting down")

    return app
