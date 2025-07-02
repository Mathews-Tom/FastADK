"""
API package for FastADK.

This module provides the FastAPI integration for FastADK.
"""

from .router import AgentRegistry, create_api_router, create_app, registry

__all__ = ["create_api_router", "create_app", "registry", "AgentRegistry"]
