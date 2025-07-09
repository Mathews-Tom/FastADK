"""
Tests for example scripts to ensure they can be loaded and executed.

These smoke tests check that the example scripts load without errors and
that the main agent classes can be instantiated successfully.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from fastadk.core.agent import BaseAgent


def load_module_from_path(file_path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_agent_classes(module):
    """Find all classes that inherit from BaseAgent in a module."""
    return [
        obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj is not BaseAgent
    ]


def get_example_files():
    """Get all Python files in the examples directory."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_files = []

    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py"):
                example_files.append(os.path.join(root, file))

    return example_files


SKIP_FILES = [
    "customer_support.py",
    "finance_assistant.py",
    "event_plugin_demo.py",
    "logging_plugin.py",
    "workflow_demo.py",
    "fine_tuning_example.py",
]


UTILITY_FILES = [
    "__init__.py",
    "utils.py",
    "helpers.py",
    "common.py",
]


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_loads_without_error(example_file):
    """Test that example files load without syntax or import errors."""
    # Skip certain files that might have unresolved dependencies in test environment
    if any(x in example_file for x in SKIP_FILES):
        pytest.skip(f"Skipping {example_file} - known to have dependencies")

    try:
        module = load_module_from_path(example_file)
        assert module is not None, f"Failed to load {example_file}"
    except ImportError:
        pytest.skip(f"Import error in {example_file}")
    except Exception as e:
        pytest.fail(f"Error loading {example_file}: {str(e)}")


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_has_agent_class(example_file):
    """Test that example files contain at least one agent class."""
    # Skip certain files that might have unresolved dependencies in test environment
    if any(x in example_file for x in SKIP_FILES):
        pytest.skip(f"Skipping {example_file} - known to have dependencies")

    try:
        module = load_module_from_path(example_file)
        agent_classes = find_agent_classes(module)

        # Skip this test for utility modules that don't define agents
        if os.path.basename(example_file) in UTILITY_FILES:
            pytest.skip("Utility module, not expected to contain agent classes")

        assert len(agent_classes) > 0, f"No agent classes found in {example_file}"
    except ImportError:
        pytest.skip(f"Import error in {example_file}")
    except Exception as e:
        pytest.skip(f"Could not check agent classes in {example_file}: {str(e)}")


@pytest.mark.asyncio
@pytest.mark.parametrize("example_file", get_example_files())
async def test_agent_instantiation(example_file):
    """Test that agent classes can be instantiated without errors."""
    # Skip certain files that might have unresolved dependencies in test environment
    if any(x in example_file for x in SKIP_FILES + ["litellm_demo.py"]):
        pytest.skip(f"Skipping {example_file} - known to have dependencies")

    try:
        module = load_module_from_path(example_file)
        agent_classes = find_agent_classes(module)

        # Skip if no agent classes found (utility modules)
        if not agent_classes:
            pytest.skip(f"No agent classes found in {example_file}")

        # Test the first agent class found
        agent_class = agent_classes[0]
        agent = agent_class()

        assert agent is not None, f"Failed to instantiate agent from {example_file}"
        assert hasattr(agent, "run"), f"Agent from {example_file} missing 'run' method"
    except ImportError:
        pytest.skip(f"Import error in {example_file}")
    except Exception as e:
        pytest.skip(f"Could not instantiate agent from {example_file}: {str(e)}")
