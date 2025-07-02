"""
Command-line interface (CLI) for the FastADK framework.

This module provides the main entry point for interacting with FastADK from the command line.
It allows users to run agents, manage projects, and access framework tools.
"""

import asyncio
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt

from fastadk.core.agent import BaseAgent

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
                except Exception as e:
                    console.print(f"\n[bold red]An error occurred:[/bold red] {e}")
                    logger.debug(f"Traceback: {e}", exc_info=True)

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
    except Exception as e:
        console.print(f"[bold red]Failed to initialize or run agent:[/bold red] {e}")
        logger.debug(f"Traceback: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """
    Display the installed version of FastADK.
    """
    from fastadk import __version__

    console.print(f"🚀 FastADK version: [bold cyan]{__version__}[/bold cyan]")


if __name__ == "__main__":
    app()
