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

import typer
import uvicorn
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.traceback import Traceback

from fastadk import __version__
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
    ValidationError,
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
        raise typer.Exit(code=1)

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


# --- CLI Commands ---


@app.command()
def run(
    module_path: Path = typer.Argument(
        ...,
        help="Path to the Python module file containing your agent class (e.g., 'my_agent.py').",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    agent_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Name of the agent class to run. If not provided, you will be prompted if multiple agents exist.",
        show_default=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose DEBUG logging for detailed output.",
    ),
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
        raise typer.Exit(code=1)

    agent_class = None
    if agent_name:
        agent_class = next((c for c in agent_classes if c.__name__ == agent_name), None)
        if not agent_class:
            console.print(
                f"[bold red]Error:[/bold red] Agent class '{agent_name}' not found in {module_path.name}."
            )
            console.print(f"Available agents: {[c.__name__ for c in agent_classes]}")
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
    try:
        agent_instance = agent_class()
        asyncio.run(_run_interactive_session(agent_instance))
    except FastADKError as e:
        handle_cli_error(e)
        raise typer.Exit(code=1)
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
        raise typer.Exit(code=1)


@app.command()
def serve(
    module_path: Path = typer.Argument(
        ...,
        help="Path to the Python module file containing your agent class(es).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="The host to bind the server to."
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="The port to bind the server to."
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload on file changes."
    ),
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
        raise typer.Exit(code=1)

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
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Failed to start server:[/bold red] {str(e)}")
        if logger.level <= logging.DEBUG:
            console.print(Traceback.from_exception(type(e), e, e.__traceback__))
        else:
            console.print(
                "[dim]Run with --verbose for detailed error information[/dim]"
            )
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """
    Display the installed version of FastADK.
    """
    console.print(f"🚀 FastADK version: [bold cyan]{__version__}[/bold cyan]")


@app.command()
def config() -> None:
    """
    Display the current configuration settings.
    """
    settings = get_settings()

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
