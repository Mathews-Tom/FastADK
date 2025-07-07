"""
Command-line interface (CLI) for the FastADK framework.

This module provides the main entry point for interacting with FastADK from the command line.
It allows users to run agents, manage projects, and access framework tools.
"""

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.traceback import Traceback

from fastadk import __version__
from fastadk.cli.repl import run_repl
from fastadk.core.agent import BaseAgent
from fastadk.core.config import get_settings
from fastadk.core.exceptions import (
    AgentError,
    ConfigurationError,
    ExceptionTracker,
    FastADKError,
    NotFoundError,
    OperationTimeoutError,
    ServiceUnavailableError,
    ToolError,
)

# --- Setup ---
# Configure logging for rich, colorful output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("fastadk")

# Initialize Typer and Rich for a modern CLI experience
app = typer.Typer(
    name="fastadk",
    help="🚀 FastADK - The developer-friendly framework for building AI agents.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,  # We use Rich for exceptions
)
console = Console()

# --- Helper Functions ---


def _find_agent_classes(module: object) -> list[type[BaseAgent]]:
    """Scans a Python module and returns a list of all classes that inherit from BaseAgent."""
    agent_classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseAgent) and obj is not BaseAgent:
            agent_classes.append(obj)
    return agent_classes


def _import_module_from_path(module_path: Path) -> object:
    """Dynamically imports a Python module from a given file path."""
    if not module_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Module file not found: {module_path}"
        )
        raise typer.Exit(code=1)

    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        console.print(
            f"[bold red]Error:[/bold red] Could not create module spec from: {module_path}"
        )
        raise typer.Exit(code=1) from None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def handle_cli_error(exc: Exception) -> None:
    """
    Format exceptions for CLI output with enhanced error information.

    Args:
        exc: The exception to format
    """
    if isinstance(exc, FastADKError):
        # Format FastADK errors with rich error display
        console.print(f"[bold red]Error [{exc.error_code}]:[/bold red] {exc.message}")

        if exc.details:
            console.print("[bold]Details:[/bold]")
            console.print(json.dumps(exc.details, indent=2))

        # Provide helpful hints based on error type
        if isinstance(exc, ConfigurationError):
            console.print(
                "[yellow]Hint:[/yellow] Check your fastadk.yaml configuration file."
            )
            console.print(
                "      Run [bold]fastadk config[/bold] to see current settings."
            )

        elif isinstance(exc, ValidationError):
            console.print(
                "[yellow]Hint:[/yellow] The input data failed validation checks."
            )

        elif isinstance(exc, ServiceUnavailableError):
            console.print(
                "[yellow]Hint:[/yellow] A required service or API is unavailable."
            )
            console.print("      Check your network connection and API keys.")

        elif isinstance(exc, OperationTimeoutError):
            console.print(
                "[yellow]Hint:[/yellow] The operation took too long to complete."
            )
            console.print(
                "      Consider increasing timeout settings or try again later."
            )

        elif isinstance(exc, ToolError):
            console.print(
                "[yellow]Hint:[/yellow] A tool executed by the agent encountered an error."
            )
            console.print("      Check the tool implementation and input data.")

        elif isinstance(exc, AgentError):
            console.print(
                "[yellow]Hint:[/yellow] The agent encountered an error during execution."
            )
            console.print("      Check agent configuration and model settings.")

        elif isinstance(exc, NotFoundError):
            console.print(
                "[yellow]Hint:[/yellow] The requested resource could not be found."
            )

    else:
        # For standard Python exceptions
        console.print(f"[bold red]Unexpected error:[/bold red] {str(exc)}")

        # Only show traceback in verbose mode
        if logger.level <= logging.DEBUG:
            console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))
        else:
            console.print(
                "[dim]Run with --verbose for detailed error information[/dim]"
            )


async def _run_interactive_session(agent: BaseAgent) -> None:
    """Handles the main interactive loop for chatting with an agent."""
    agent_name = agent.__class__.__name__
    console.print(
        Panel.fit(
            f"[bold]Entering interactive session with [cyan]{agent_name}[/cyan][/bold]\n"
            f"Type 'exit' or 'quit', or press Ctrl+D to end.",
            title="⚡️ FastADK Live",
            border_style="blue",
        )
    )

    session_id = 1
    try:
        while True:
            prompt = Prompt.ask(f"\n[bold blue]You (session {session_id})[/bold blue]")
            if prompt.lower() in ("exit", "quit"):
                break

            with console.status(
                "[bold green]Agent is thinking...[/bold green]", spinner="dots"
            ):
                try:
                    response = await agent.run(prompt)
                    console.print(f"\n[bold green]Agent[/bold green]: {response}")
                except FastADKError as e:
                    handle_cli_error(e)
                except Exception as e:
                    console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
                    if logger.level <= logging.DEBUG:
                        console.print(
                            Traceback.from_exception(type(e), e, e.__traceback__)
                        )

            session_id += 1

    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C and Ctrl+D gracefully
        pass
    finally:
        console.print("\n\n[italic]Interactive session ended. Goodbye![/italic]")


# --- Define Module-Level Option Variables ---

# Options for the run command
MODULE_PATH_ARG = typer.Argument(
    ...,
    help="Path to the Python module file containing your agent class (e.g., 'my_agent.py').",
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)

AGENT_NAME_OPT = typer.Option(
    None,
    "--name",
    "-n",
    help="Name of the agent class to run. If not provided, you will be prompted if multiple agents exist.",
    show_default=False,
)

VERBOSE_OPT = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose DEBUG logging for detailed output.",
)

# --- CLI Commands ---


@app.command()
def run(
    module_path: Path = MODULE_PATH_ARG,
    agent_name: str | None = AGENT_NAME_OPT,
    verbose: bool = VERBOSE_OPT,
) -> None:
    """
    Run an agent in an interactive command-line chat session.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        console.print("[yellow]Verbose logging enabled.[/yellow]")

    module = _import_module_from_path(module_path)
    agent_classes = _find_agent_classes(module)

    if not agent_classes:
        console.print(
            f"[bold red]Error:[/bold red] No FastADK agent classes found in {module_path.name}."
        )
        raise typer.Exit(code=1) from None

    agent_class = None
    if agent_name:
        agent_class = next((c for c in agent_classes if c.__name__ == agent_name), None)
        if not agent_class:
            console.print(
                f"[bold red]Error:[/bold red] Agent class '{agent_name}' not found in {module_path.name}."
            )
            console.print(f"Available agents: {[c.__name__ for c in agent_classes]}")
            raise typer.Exit(code=1) from None
    elif len(agent_classes) == 1:
        agent_class = agent_classes[0]
    else:
        # Prompt user to choose if multiple agents are found
        choices = {str(i + 1): c for i, c in enumerate(agent_classes)}
        console.print(
            "[bold yellow]Multiple agents found. Please choose one:[/bold yellow]"
        )
        for i, c in choices.items():
            console.print(f"  [cyan]{i}[/cyan]: {c.__name__}")

        choice = Prompt.ask(
            "Enter the number of the agent to run",
            choices=list(choices.keys()),
            default="1",
        )
        agent_class = choices[choice]

    console.print(
        f"Initializing agent: [bold cyan]{agent_class.__name__}[/bold cyan]..."
    )
    try:
        agent_instance = agent_class()
        asyncio.run(_run_interactive_session(agent_instance))
    except FastADKError as e:
        handle_cli_error(e)
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(
            f"[bold red]Failed to initialize or run agent:[/bold red] {str(e)}"
        )
        if logger.level <= logging.DEBUG:
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        else:
            console.print(
                "[dim]Run with --verbose for detailed error information[/dim]"
            )
        raise typer.Exit(code=1) from None


# Options for the serve command
SERVE_MODULE_PATH_ARG = typer.Argument(
    ...,
    help="Path to the Python module file containing your agent class(es).",
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)

HOST_OPT = typer.Option(
    "127.0.0.1", "--host", "-h", help="The host to bind the server to."
)

PORT_OPT = typer.Option(8000, "--port", "-p", help="The port to bind the server to.")

RELOAD_OPT = typer.Option(False, "--reload", help="Enable auto-reload on file changes.")


@app.command()
def serve(
    module_path: Path = SERVE_MODULE_PATH_ARG,
    host: str = HOST_OPT,
    port: int = PORT_OPT,
    reload: bool = RELOAD_OPT,
) -> None:
    """
    Start an HTTP server to serve your agents via a REST API.
    """
    console.print(
        Panel.fit(
            "Starting FastADK API server...\n"
            f"Loading agents from: [cyan]{module_path}[/cyan]",
            title="🚀 FastADK API",
            border_style="green",
        )
    )

    # Import module and find agent classes
    module = _import_module_from_path(module_path)
    agent_classes = _find_agent_classes(module)

    if not agent_classes:
        console.print(
            f"[bold red]Error:[/bold red] No FastADK agent classes found in {module_path.name}."
        )
        raise typer.Exit(code=1) from None

    # Import API components here to avoid circular imports
    from fastadk.api.router import create_app, registry

    # Register all agents found in the module
    for agent_class in agent_classes:
        registry.register(agent_class)
        console.print(
            f"  - Registered agent: [bold cyan]{agent_class.__name__}[/bold cyan]"
        )

    # Create a table with registered agents and their endpoints
    table = Table(title="Available API Endpoints")
    table.add_column("Agent", style="cyan")
    table.add_column("Endpoint", style="green")
    table.add_column("Description", style="white")

    for agent_class in agent_classes:
        # Use getattr for _description to avoid mypy error with protected member access
        description = getattr(agent_class, "_description", "")
        table.add_row(
            agent_class.__name__,
            f"POST /agents/{agent_class.__name__}",
            description,
        )

    console.print(table)
    console.print(
        f"\nAPI Documentation: [bold blue]http://{host}:{port}/docs[/bold blue]"
    )

    # Set environment variable to identify the module path for use in reload mode
    os.environ["FASTADK_MODULE_PATH"] = str(module_path)

    # Start Uvicorn server
    try:
        # Create the app with our registry and pass it to uvicorn
        api_app = create_app()
        uvicorn.run(
            api_app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except FastADKError as e:
        handle_cli_error(e)
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[bold red]Failed to start server:[/bold red] {str(e)}")
        if logger.level <= logging.DEBUG:
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        else:
            console.print(
                "[dim]Run with --verbose for detailed error information[/dim]"
            )
        raise typer.Exit(code=1) from None


# Options for the repl command
REPL_MODULE_PATH_OPT = typer.Option(
    None,
    "--module",
    "-m",
    help="Path to the Python module file containing your agent class.",
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)

REPL_AGENT_NAME_OPT = typer.Option(
    None,
    "--name",
    "-n",
    help="Name of the agent class to run. If not provided, you will be prompted if multiple agents exist.",
)


@app.command()
def repl(
    module_path: Optional[Path] = REPL_MODULE_PATH_OPT,
    agent_name: str = REPL_AGENT_NAME_OPT,
    verbose: bool = VERBOSE_OPT,
) -> None:
    """
    Start an interactive REPL session with an agent.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        console.print("[yellow]Verbose logging enabled.[/yellow]")

    # If no module_path is provided, start a REPL with a default agent
    if not module_path:
        console.print("[yellow]No agent module specified, using simple agent.[/yellow]")
        # This would normally create a simple default agent
        from fastadk.core.agent import BaseAgent

        class SimpleAgent(BaseAgent):
            """A simple agent for demo purposes."""

            _description = "A simple demo agent"
            _model_name = "gpt-3.5-turbo"
            _provider = "openai"

            async def run(self, prompt: str) -> str:
                """Simple implementation that echoes the prompt."""
                return f"You said: {prompt}\n\nThis is a demo agent. Specify a real agent with --module."

        agent = SimpleAgent()
        asyncio.run(run_repl(agent))
        return

    # Otherwise, load the specified module and agent
    try:
        module = _import_module_from_path(module_path)
        agent_classes = _find_agent_classes(module)

        if not agent_classes:
            console.print(
                f"[bold red]Error:[/bold red] No FastADK agent classes found in {module_path.name}."
            )
            raise typer.Exit(code=1)

        agent_class = None
        if agent_name:
            agent_class = next(
                (c for c in agent_classes if c.__name__ == agent_name), None
            )
            if not agent_class:
                console.print(
                    f"[bold red]Error:[/bold red] Agent class '{agent_name}' not found in {module_path.name}."
                )
                console.print(
                    f"Available agents: {[c.__name__ for c in agent_classes]}"
                )
                raise typer.Exit(code=1)
        elif len(agent_classes) == 1:
            agent_class = agent_classes[0]
        else:
            # Prompt user to choose if multiple agents are found
            choices = {str(i + 1): c for i, c in enumerate(agent_classes)}
            console.print(
                "[bold yellow]Multiple agents found. Please choose one:[/bold yellow]"
            )
            for i, c in choices.items():
                console.print(f"  [cyan]{i}[/cyan]: {c.__name__}")

            choice = Prompt.ask(
                "Enter the number of the agent to run",
                choices=list(choices.keys()),
                default="1",
            )
            agent_class = choices[choice]

        console.print(
            f"Initializing agent: [bold cyan]{agent_class.__name__}[/bold cyan]..."
        )
        agent_instance = agent_class()
        asyncio.run(run_repl(agent_instance))
    except FastADKError as e:
        handle_cli_error(e)
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(
            f"[bold red]Failed to initialize or run agent:[/bold red] {str(e)}"
        )
        if logger.level <= logging.DEBUG:
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        else:
            console.print(
                "[dim]Run with --verbose for detailed error information[/dim]"
            )
        raise typer.Exit(code=1) from None


# Options for the init command
PROJECT_NAME_ARG = typer.Argument(..., help="Name of the project to create")

DIRECTORY_OPT = typer.Option(
    None,
    "--dir",
    "-d",
    help="Directory to create the project in. Defaults to a directory with the project name.",
)

TEMPLATE_OPT = typer.Option(
    "basic",
    "--template",
    "-t",
    help="Project template to use (basic, advanced, api)",
)


@app.command()
def init(
    project_name: str = PROJECT_NAME_ARG,
    directory: Path = DIRECTORY_OPT,
    template: str = TEMPLATE_OPT,
) -> None:
    """
    Initialize a new FastADK project with scaffolding.
    """
    templates = ["basic", "advanced", "api"]
    if template not in templates:
        console.print(
            f"[bold red]Error:[/bold red] Unknown template: {template}. Available templates: {', '.join(templates)}"
        )
        raise typer.Exit(code=1) from None

    # Determine the project directory
    if directory is None:
        directory = Path(project_name)

    # Check if the directory already exists
    if directory.exists():
        if not directory.is_dir():
            console.print(
                f"[bold red]Error:[/bold red] {directory} exists but is not a directory."
            )
            raise typer.Exit(code=1) from None

        # Check if the directory is empty
        if any(directory.iterdir()):
            overwrite = Prompt.ask(
                f"Directory {directory} is not empty. Continue anyway?",
                choices=["y", "n"],
                default="n",
            )
            if overwrite.lower() != "y":
                console.print("Project initialization cancelled.")
                raise typer.Exit(code=0) from None
    else:
        # Create the directory
        directory.mkdir(parents=True)

    # Create a basic project structure
    console.print(
        f"[bold green]Creating {template} project in {directory}[/bold green]"
    )

    # Create the project files based on template
    if template == "basic":
        # Create basic directory structure
        (directory / "agents").mkdir(exist_ok=True)
        (directory / "tools").mkdir(exist_ok=True)
        (directory / "data").mkdir(exist_ok=True)

        # Create main agent file
        agent_file = directory / "agents" / "my_agent.py"
        agent_code = """
from fastadk.core.agent import BaseAgent

class MyAgent(BaseAgent):
    \"\"\"My custom agent.\"\"\"
    _description = "A basic FastADK agent"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    async def run(self, prompt: str) -> str:
        \"\"\"Run the agent with the given prompt.\"\"\"
        # This is a simple implementation that uses the base Agent's functionality
        return await super().run(prompt)
"""
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write(agent_code.lstrip())

        # Create README file
        readme = directory / "README.md"
        readme_content = f"""
# {project_name}

A FastADK project created with the basic template.

## Getting Started

1. Install FastADK:
   ```
   pip install fastadk
   ```

2. Set up your environment variables:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run your agent:
   ```
   fastadk run agents/my_agent.py
   ```

## Project Structure

- `agents/`: Contains your agent classes
- `tools/`: Custom tools for your agents
- `data/`: Data files used by your agents
"""
        with open(readme, "w", encoding="utf-8") as f:
            f.write(readme_content.lstrip())

        # Create config file
        config_file = directory / "fastadk.yaml"
        config_content = """
# FastADK Configuration
environment: development

model:
  provider: openai
  model_name: gpt-3.5-turbo
  api_key_env_var: OPENAI_API_KEY

memory:
  backend_type: inmemory
  ttl_seconds: 3600

telemetry:
  enabled: true
  log_level: info

security:
  content_filtering: true
  pii_detection: false
  audit_logging: false
"""
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content.lstrip())

    elif template == "advanced":
        # Create more advanced structure for larger projects
        (directory / "src" / project_name.lower() / "agents").mkdir(
            parents=True, exist_ok=True
        )
        (directory / "src" / project_name.lower() / "tools").mkdir(
            parents=True, exist_ok=True
        )
        (directory / "src" / project_name.lower() / "utils").mkdir(
            parents=True, exist_ok=True
        )
        (directory / "tests").mkdir(exist_ok=True)
        (directory / "data").mkdir(exist_ok=True)

        # Create __init__.py files
        with open(
            directory / "src" / project_name.lower() / "__init__.py",
            "w",
            encoding="utf-8",
        ) as f:
            f.write('"""Main package for the project."""\n')

        with open(
            directory / "src" / project_name.lower() / "agents" / "__init__.py",
            "w",
            encoding="utf-8",
        ) as f:
            f.write('"""Agent implementations."""\n')

        with open(
            directory / "src" / project_name.lower() / "tools" / "__init__.py",
            "w",
            encoding="utf-8",
        ) as f:
            f.write('"""Custom tools for agents."""\n')

        with open(
            directory / "src" / project_name.lower() / "utils" / "__init__.py",
            "w",
            encoding="utf-8",
        ) as f:
            f.write('"""Utility functions."""\n')

        # Create main agent file
        agent_file = (
            directory / "src" / project_name.lower() / "agents" / "main_agent.py"
        )
        agent_code = """
from fastadk.core.agent import BaseAgent
from fastadk.core.context import Context

class MainAgent(BaseAgent):
    \"\"\"Main agent for the application.\"\"\"
    _description = "An advanced FastADK agent with context management"
    _model_name = "gpt-4"
    _provider = "openai"
    
    async def run(self, prompt: str) -> str:
        \"\"\"Run the agent with the given prompt.\"\"\"
        # Initialize context
        context = Context()
        context.add("prompt", prompt)
        
        # Process the prompt
        response = await self.model.generate(
            f"You are a helpful assistant. Respond to the following: {prompt}"
        )
        
        return response
"""
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write(agent_code.lstrip())

        # Create pyproject.toml
        pyproject = directory / "pyproject.toml"
        pyproject_content = f"""
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{project_name.lower()}"
version = "0.1.0"
authors = [
    {{name = "Your Name", email = "your.email@example.com"}},
]
description = "A FastADK project"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "fastadk>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio",
    "ruff",
    "mypy",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
"""
        with open(pyproject, "w", encoding="utf-8") as f:
            f.write(pyproject_content.lstrip())

        # Create README file
        readme = directory / "README.md"
        readme_content = f"""
# {project_name}

A FastADK project created with the advanced template.

## Getting Started

1. Install the project:
   ```
   pip install -e .
   ```

2. Set up your environment variables:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run your agent:
   ```
   fastadk run src/{project_name.lower()}/agents/main_agent.py
   ```

## Project Structure

- `src/{project_name.lower()}/agents/`: Contains your agent classes
- `src/{project_name.lower()}/tools/`: Custom tools for your agents
- `src/{project_name.lower()}/utils/`: Utility functions
- `tests/`: Unit tests
- `data/`: Data files used by your agents
"""
        with open(readme, "w", encoding="utf-8") as f:
            f.write(readme_content.lstrip())

    elif template == "api":
        # Create API-focused project
        (directory / "agents").mkdir(exist_ok=True)
        (directory / "api").mkdir(exist_ok=True)
        (directory / "tools").mkdir(exist_ok=True)

        # Create main agent file
        agent_file = directory / "agents" / "api_agent.py"
        agent_code = """
from fastadk.core.agent import BaseAgent

class ApiAgent(BaseAgent):
    \"\"\"Agent for API integration.\"\"\"
    _description = "An agent designed for API deployment"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"
    
    async def run(self, prompt: str) -> str:
        \"\"\"Run the agent with the given prompt.\"\"\"
        # Process the prompt
        response = await self.model.generate(
            f"You are a helpful API assistant. Respond to the following: {prompt}"
        )
        
        return response
"""
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write(agent_code.lstrip())

        # Create API runner
        api_file = directory / "api" / "main.py"
        api_code = """
import uvicorn

from fastadk.api.router import create_app, registry
from agents.api_agent import ApiAgent

# Register the agent
registry.register(ApiAgent)

# Create the FastAPI app
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open(api_file, "w", encoding="utf-8") as f:
            f.write(api_code.lstrip())

        # Create README file
        readme = directory / "README.md"
        readme_content = f"""
# {project_name}

A FastADK API project created with the API template.

## Getting Started

1. Install FastADK and dependencies:
   ```
   pip install fastadk uvicorn
   ```

2. Set up your environment variables:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run the API server:
   ```
   python api/main.py
   ```
   
   Or use the FastADK CLI:
   ```
   fastadk serve agents/api_agent.py
   ```

4. Open your browser to http://localhost:8000/docs to see the API documentation.

## Project Structure

- `agents/`: Contains your agent classes
- `api/`: API server setup
- `tools/`: Custom tools for your agents
"""
        with open(readme, "w", encoding="utf-8") as f:
            f.write(readme_content.lstrip())

    console.print(
        f"[bold green]✅ Project created successfully in {directory}[/bold green]"
    )


@app.command()
def version() -> None:
    """
    Display the installed version of FastADK.
    """
    console.print(f"🚀 FastADK version: [bold cyan]{__version__}[/bold cyan]")


# Options for the config command
CONFIG_VALIDATE_OPT = typer.Option(
    False, "--validate", "-v", help="Validate the configuration file."
)

CONFIG_PATH_OPT = typer.Option(
    None, "--path", "-p", help="Path to a specific configuration file to validate."
)


@app.command()
def config(
    validate: bool = CONFIG_VALIDATE_OPT,
    path: Optional[Path] = CONFIG_PATH_OPT,
) -> None:
    """
    Display or validate the current configuration settings.
    """
    settings = get_settings()

    if validate:
        # Validate the configuration
        try:
            if path:
                # Validate a specific configuration file
                if not path.exists():
                    console.print(
                        f"[bold red]Error:[/bold red] Configuration file not found: {path}"
                    )
                    raise typer.Exit(code=1)

                # Try to load and validate the config
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        if path.suffix.lower() == ".json":
                            config_data = json.load(f)
                        else:
                            # Simple format check, not actual validation
                            config_data = {}
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    key = line.split(":", 1)[0].strip()
                                    config_data[key] = True

                    # This would normally validate the config
                    # For now we just check if essential keys are present
                    required_keys = ["model", "memory", "telemetry", "security"]
                    missing_keys = [
                        key for key in required_keys if key not in config_data
                    ]

                    if missing_keys:
                        raise ValidationError(
                            f"Missing required configuration sections: {', '.join(missing_keys)}",
                            model=None,
                        )
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ValidationError(
                        "Invalid configuration file format",
                        model=None,
                    ) from e

                console.print(
                    f"[bold green]Success:[/bold green] Configuration file is valid: {path}"
                )
            else:
                # Validate the current configuration
                console.print(
                    f"[bold green]Success:[/bold green] Current configuration is valid."
                )

                if settings.config_path:
                    console.print(
                        f"[dim]Config loaded from: {settings.config_path}[/dim]"
                    )
        except ValidationError as e:
            console.print(f"[bold red]Configuration validation failed:[/bold red]")
            console.print(f"[red]{str(e)}[/red]")
            if e.details:
                console.print("\n[bold]Details:[/bold]")
                console.print(json.dumps(e.details, indent=2))
            raise typer.Exit(code=1) from None
        except Exception as e:
            console.print(
                f"[bold red]Error validating configuration:[/bold red] {str(e)}"
            )
            raise typer.Exit(code=1) from None

        return

    # Display configuration
    console.print(
        Panel(
            "[bold]FastADK Configuration[/bold]\n",
            title="⚙️ Settings",
            border_style="blue",
        )
    )

    # Environment
    console.print(f"[bold]Environment:[/bold] [cyan]{settings.environment}[/cyan]")

    # Model configuration
    console.print("\n[bold]Model Configuration:[/bold]")
    # Access attributes directly with fallbacks
    provider = getattr(settings.model, "provider", "unknown")
    model_name = getattr(settings.model, "model_name", "unknown")
    api_key_var = getattr(settings.model, "api_key_env_var", "unknown")

    console.print(f"  Provider: [cyan]{provider}[/cyan]")
    console.print(f"  Model: [cyan]{model_name}[/cyan]")
    console.print(f"  API Key Env Var: [cyan]{api_key_var}[/cyan]")

    # Memory configuration
    console.print("\n[bold]Memory Configuration:[/bold]")
    console.print(f"  Backend: [cyan]{settings.memory.backend_type}[/cyan]")
    console.print(f"  TTL: [cyan]{settings.memory.ttl_seconds}s[/cyan]")

    # Telemetry
    console.print("\n[bold]Telemetry Configuration:[/bold]")
    console.print(f"  Enabled: [cyan]{settings.telemetry.enabled}[/cyan]")
    console.print(f"  Log Level: [cyan]{settings.telemetry.log_level}[/cyan]")

    # Security
    console.print("\n[bold]Security Configuration:[/bold]")
    console.print(
        f"  Content Filtering: [cyan]{settings.security.content_filtering}[/cyan]"
    )
    console.print(f"  PII Detection: [cyan]{settings.security.pii_detection}[/cyan]")
    console.print(f"  Audit Logging: [cyan]{settings.security.audit_logging}[/cyan]")

    # Config paths
    if settings.config_path:
        console.print(
            f"\n[bold]Config loaded from:[/bold] [green]{settings.config_path}[/green]"
        )
    else:
        console.print(
            "\n[bold yellow]No config file found. Using defaults and environment variables.[/bold yellow]"
        )


@app.command()
def errors(
    sample: bool = typer.Option(
        False, "--sample", "-s", help="Generate a sample error for testing"
    ),
) -> None:
    """
    Display error statistics and recent exceptions.
    """
    # Generate a sample error if requested
    if sample:
        console.print("[yellow]Generating sample errors for testing...[/yellow]")
        try:
            # Sample validation error
            raise ValidationError(
                message="Sample validation error",
                error_code="SAMPLE_VALIDATION_ERROR",
                details={"sample": True, "value": "test"},
            )
        except FastADKError as e:
            ExceptionTracker.track_exception(e)

        try:
            # Sample configuration error
            raise ConfigurationError(
                message="Sample configuration error",
                error_code="SAMPLE_CONFIG_ERROR",
                details={"setting": "api_key", "required": True},
            )
        except FastADKError as e:
            ExceptionTracker.track_exception(e)

        try:
            # Sample service error
            raise ServiceUnavailableError(
                message="Sample service unavailable",
                error_code="SAMPLE_SERVICE_ERROR",
                details={"service": "external_api", "status": 503},
            )
        except FastADKError as e:
            ExceptionTracker.track_exception(e)
    summary = ExceptionTracker.get_summary()
    recent = ExceptionTracker.get_recent_exceptions(limit=5)

    console.print(
        Panel(
            "[bold]FastADK Error Statistics[/bold]\n",
            title="🛑 Errors",
            border_style="red",
        )
    )

    # Summary section
    console.print("[bold]Error Summary:[/bold]")
    console.print(f"  Total Exceptions: [cyan]{summary['total_exceptions']}[/cyan]")
    console.print(f"  Unique Error Codes: [cyan]{summary['unique_error_codes']}[/cyan]")

    if summary.get("tracked_period_seconds"):
        period = summary["tracked_period_seconds"]
        console.print(f"  Tracking Period: [cyan]{period:.1f} seconds[/cyan]")

    # Top errors
    if summary.get("top_errors"):
        console.print("\n[bold]Top Error Types:[/bold]")
        for code, count in summary["top_errors"].items():
            console.print(f"  [red]{code}[/red]: {count} occurrences")

    # Recent errors table
    if recent:
        console.print("\n[bold]Recent Exceptions:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Error Code", style="red")
        table.add_column("Message")
        table.add_column("Type", style="cyan")

        for exc in recent:
            table.add_row(
                exc["error_code"] or "UNKNOWN",
                exc["message"][:50] + ("..." if len(exc["message"]) > 50 else ""),
                exc["exception_type"],
            )

        console.print(table)
    else:
        console.print("\n[green]No exceptions tracked yet.[/green]")

    console.print(
        "\n[italic]Use the [bold]--verbose[/bold] flag with commands to see detailed error information.[/italic]"
    )


if __name__ == "__main__":
    app()
