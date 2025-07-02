"""
Testing utilities for FastADK agents.

This module provides testing utilities and fixtures for testing FastADK agents.
"""

from collections.abc import Callable
from typing import Any

# To avoid circular imports, importing these classes needs to be
# done carefully when they're needed
__all__ = [
    "AgentTest",
    "MockModel",
    "load_test",
    "scenario",
]

# Create placeholders that will be replaced with real implementations when imported
AgentTest = object
MockModel = object


def load_test(*_args: object, **_kwargs: object) -> Any:
    """Placeholder for the load_test function."""
    return None


def scenario(*_args: object, **_kwargs: object) -> Any:
    """Placeholder for the scenario function."""
    return lambda f: f


# Removed test_scenario as it doesn't exist in utils.py


# Make these available when explicitly imported
def __getattr__(name: str) -> object:
    """Lazily import testing utilities to avoid circular imports."""
    if name in __all__:
        # Only import when actually used
        from .utils import AgentTest as _AgentTest
        from .utils import MockModel as _MockModel
        from .utils import load_test as _load_test
        from .utils import scenario as _scenario

        # Update the module globals
        global AgentTest, MockModel, load_test, scenario
        AgentTest = _AgentTest
        MockModel = _MockModel
        load_test = _load_test  # type: ignore
        scenario = _scenario  # type: ignore

        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
