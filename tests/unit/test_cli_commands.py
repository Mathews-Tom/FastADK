"""
Unit tests for FastADK CLI commands.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fastadk.cli.main import app
from fastadk.cli.repl import ReplEnvironment


@pytest.fixture
def cli_runner():
    """Fixture for testing CLI commands."""
    return CliRunner()


def test_version(cli_runner):
    """Test the version command."""
    from fastadk import __version__

    result = cli_runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_display(cli_runner):
    """Test the config command (display mode)."""
    result = cli_runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "FastADK Configuration" in result.stdout
    assert "Environment:" in result.stdout
    assert "Model Configuration:" in result.stdout


@patch("pathlib.Path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_config_validate_success(mock_open, mock_exists, cli_runner):
    """Test the config validation command (success case)."""
    # Setup mocks
    mock_exists.return_value = True
    mock_open.return_value.__enter__.return_value.read.return_value = (
        '{"model": {}, "memory": {}, "telemetry": {}, "security": {}}'
    )

    # Run the command
    result = cli_runner.invoke(
        app, ["config", "--validate", "--path", "test_config.json"]
    )

    # Check the result
    assert result.exit_code == 0
    assert "Success" in result.stdout


@patch("pathlib.Path.exists")
def test_config_validate_file_not_found(mock_exists, cli_runner):
    """Test the config validation command (file not found)."""
    # Setup mocks
    mock_exists.return_value = False

    # Run the command
    result = cli_runner.invoke(
        app, ["config", "--validate", "--path", "non_existent.json"]
    )

    # Check the result
    assert result.exit_code == 1
    assert "Error" in result.stdout
    assert "not found" in result.stdout


def test_repl_environment():
    """Test the REPL environment class."""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_agent.__class__.__name__ = "MockAgent"
    mock_agent._description = "A mock agent for testing"
    mock_agent._model_name = "mock-model"
    mock_agent._provider = "mock-provider"
    mock_agent.tools = {"test_tool": MagicMock(description="A test tool")}

    # Create a REPL environment
    env = ReplEnvironment(mock_agent)

    # Test basic properties
    assert env.agent == mock_agent
    assert not env.verbose
    assert env.markdown_output
    assert env.token_count

    # Test toggling settings
    assert env.toggle_verbose() is True
    assert env.verbose is True

    assert env.toggle_markdown() is False
    assert env.markdown_output is False

    # Test agent info
    info = env.get_agent_info()
    assert info["name"] == "MockAgent"
    assert info["description"] == "A mock agent for testing"
    assert info["model"] == "mock-model"
    assert info["provider"] == "mock-provider"
    assert "test_tool" in info["tools"]

    # Test history management
    assert len(env.history) == 0
    env.clear_history()
    assert len(env.history) == 0


def test_init_command(cli_runner):
    """Test the init command."""
    with TemporaryDirectory() as temp_dir:
        # Convert to Path object
        temp_path = Path(temp_dir)
        project_dir = temp_path / "test_project"

        # Run the init command
        result = cli_runner.invoke(
            app,
            ["init", "test_project", "--dir", str(project_dir), "--template", "basic"],
        )

        # Check the result
        assert result.exit_code == 0
        assert "Creating basic project" in result.stdout

        # Check that the files were created
        assert (project_dir / "agents").exists()
        assert (project_dir / "agents" / "my_agent.py").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "fastadk.yaml").exists()


def test_init_command_with_existing_dir(cli_runner):
    """Test the init command with an existing directory."""
    with TemporaryDirectory() as temp_dir:
        # Convert to Path object
        temp_path = Path(temp_dir)

        # Create a file in the directory
        with open(temp_path / "existing_file.txt", "w", encoding="utf-8") as f:
            f.write("This is an existing file")

        # Run the init command with the "n" response to the prompt
        result = cli_runner.invoke(
            app,
            ["init", "test_project", "--dir", str(temp_path), "--template", "basic"],
            input="n\n",
        )

        # Check the result - just make sure it contains "cancelled"
        assert "cancelled" in result.stdout
