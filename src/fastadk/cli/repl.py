"""
Interactive REPL (Read-Eval-Print-Loop) for FastADK.

This module provides an interactive shell for working with FastADK agents,
making it easier to test and debug agent behavior.
"""

import asyncio
import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

import aioconsole
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from fastadk.core.agent import BaseAgent
from fastadk.core.config import get_settings
from fastadk.core.exceptions import FastADKError
from fastadk.tokens.counting import count_tokens

console = Console()


class ReplEnvironment:
    """
    Environment for the FastADK REPL.

    Manages the agent instance, history, and provides utility functions
    for the interactive REPL.
    """

    def __init__(self, agent: BaseAgent) -> None:
        """
        Initialize the REPL environment with an agent instance.

        Args:
            agent: The agent instance to interact with
        """
        self.agent = agent
        self.history: List[Dict[str, Any]] = []
        self.verbose = False
        self.markdown_output = True
        self.token_count = True

    async def run_prompt(self, prompt: str) -> str:
        """
        Run a prompt through the agent and capture the response.

        Args:
            prompt: User input prompt

        Returns:
            The agent's response
        """
        try:
            # Log the user input in verbose mode
            if self.verbose:
                console.print("\n[dim]Agent received prompt:[/dim]")
                console.print(Syntax(prompt, "text", theme="monokai"))

            # Run the agent
            start_token_count = None
            if self.token_count:
                start_token_count = (
                    self.agent._token_usage.total_tokens
                    if hasattr(self.agent, "_token_usage")
                    else None
                )

            response = await self.agent.run(prompt)

            # Track in history
            entry = {
                "role": "user",
                "content": prompt,
                "timestamp": "2023-07-07T12:00:00Z",  # Would use real timestamp in production
            }
            self.history.append(entry)

            # Add response to history
            entry = {
                "role": "assistant",
                "content": response,
                "timestamp": "2023-07-07T12:00:00Z",  # Would use real timestamp in production
            }

            if hasattr(self.agent, "tools_used") and self.agent.tools_used:
                entry["tools_used"] = self.agent.tools_used

            self.history.append(entry)

            # Show token usage in verbose mode
            if (
                self.verbose
                and self.token_count
                and hasattr(self.agent, "_token_usage")
            ):
                end_token_count = self.agent._token_usage.total_tokens
                if start_token_count is not None:
                    tokens_used = end_token_count - start_token_count
                    console.print(f"\n[dim]Tokens used: {tokens_used}[/dim]")
                    console.print(f"[dim]Total tokens: {end_token_count}[/dim]")

            return response

        except FastADKError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e.message}")
            if hasattr(e, "details") and e.details:
                console.print("[bold]Details:[/bold]")
                console.print(json.dumps(e.details, indent=2))
            return f"Error: {e.message}"
        except Exception as e:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
            return f"Unexpected error: {str(e)}"

    def display_response(self, response: str) -> None:
        """
        Display the agent's response with appropriate formatting.

        Args:
            response: The response to display
        """
        if self.markdown_output and not response.startswith("Error:"):
            # Try to render as markdown if enabled
            try:
                md = Markdown(response)
                console.print(Panel(md, title="Agent Response", border_style="green"))
            except Exception:
                # Fall back to plain text if markdown rendering fails
                console.print(
                    Panel(response, title="Agent Response", border_style="green")
                )
        else:
            # Plain text output
            console.print(Panel(response, title="Agent Response", border_style="green"))

    def toggle_verbose(self) -> bool:
        """Toggle verbose mode."""
        self.verbose = not self.verbose
        return self.verbose

    def toggle_markdown(self) -> bool:
        """Toggle markdown rendering for responses."""
        self.markdown_output = not self.markdown_output
        return self.markdown_output

    def toggle_token_count(self) -> bool:
        """Toggle token counting display."""
        self.token_count = not self.token_count
        return self.token_count

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent."""
        return {
            "name": self.agent.__class__.__name__,
            "description": getattr(
                self.agent, "_description", "No description available"
            ),
            "model": getattr(self.agent, "_model_name", "Unknown"),
            "provider": getattr(self.agent, "_provider", "Unknown"),
            "tools": list(getattr(self.agent, "tools", {}).keys()),
            "memory_backend": (
                getattr(self.agent, "memory_backend", "Unknown").__class__.__name__
                if hasattr(self.agent, "memory_backend")
                else "None"
            ),
        }

    def export_history(self, path: str) -> bool:
        """
        Export conversation history to a JSON file.

        Args:
            path: Path to save the history

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            console.print(f"[bold red]Error exporting history:[/bold red] {str(e)}")
            return False

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
        if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "clear"):
            self.agent.memory.clear()
        console.print("[green]Conversation history cleared.[/green]")


async def run_repl(agent: BaseAgent, welcome_message: Optional[str] = None) -> None:
    """
    Start an interactive REPL session with the agent.

    Args:
        agent: The agent instance to interact with
        welcome_message: Optional custom welcome message
    """
    env = ReplEnvironment(agent)

    # Print welcome message
    if not welcome_message:
        welcome_message = f"""
🚀 FastADK Interactive REPL (v{get_settings().version})
Agent: [bold cyan]{agent.__class__.__name__}[/bold cyan]

Special commands:
  .help          Show this help message
  .verbose       Toggle verbose output mode
  .markdown      Toggle markdown rendering of responses
  .tokens        Toggle token counting display
  .info          Show information about the current agent
  .tools         List available tools
  .history       Show conversation history
  .export PATH   Export conversation history to a file
  .clear         Clear conversation history
  .quit or .exit Exit the REPL

Type your message to interact with the agent.
"""

    console.print(
        Panel(welcome_message, title="Welcome to FastADK REPL", border_style="blue")
    )

    # Main REPL loop
    try:
        while True:
            # Get user input
            user_input = await aioconsole.ainput("\n[bold blue]>>> [/bold blue]")

            # Check for commands
            if not user_input or user_input.strip() == "":
                continue

            if user_input.startswith("."):
                command = user_input[1:].strip().lower()

                if command in ("quit", "exit"):
                    console.print("\n[italic]Goodbye![/italic]")
                    break

                elif command == "help":
                    console.print(
                        Panel(
                            welcome_message,
                            title="FastADK REPL Help",
                            border_style="blue",
                        )
                    )

                elif command == "verbose":
                    verbose = env.toggle_verbose()
                    console.print(
                        f"[green]Verbose mode {'enabled' if verbose else 'disabled'}[/green]"
                    )

                elif command == "markdown":
                    markdown = env.toggle_markdown()
                    console.print(
                        f"[green]Markdown rendering {'enabled' if markdown else 'disabled'}[/green]"
                    )

                elif command == "tokens":
                    tokens = env.toggle_token_count()
                    console.print(
                        f"[green]Token counting {'enabled' if tokens else 'disabled'}[/green]"
                    )

                elif command == "info":
                    info = env.get_agent_info()
                    console.print(
                        Panel(
                            f"Name: [bold]{info['name']}[/bold]\n"
                            f"Description: {info['description']}\n"
                            f"Model: {info['model']}\n"
                            f"Provider: {info['provider']}\n"
                            f"Memory Backend: {info['memory_backend']}\n"
                            f"Tools: {', '.join(info['tools']) if info['tools'] else 'None'}",
                            title="Agent Information",
                            border_style="yellow",
                        )
                    )

                elif command == "tools":
                    if hasattr(agent, "tools") and agent.tools:
                        tool_info = []
                        for name, tool in agent.tools.items():
                            description = getattr(
                                tool, "description", "No description available"
                            )
                            tool_info.append(f"[bold]{name}[/bold]: {description}")

                        console.print(
                            Panel(
                                "\n".join(tool_info),
                                title="Available Tools",
                                border_style="yellow",
                            )
                        )
                    else:
                        console.print(
                            "[yellow]No tools available for this agent.[/yellow]"
                        )

                elif command == "history":
                    if not env.history:
                        console.print("[yellow]No conversation history yet.[/yellow]")
                        continue

                    for entry in env.history:
                        role = entry["role"]
                        content = entry["content"]

                        if role == "user":
                            console.print(f"\n[bold blue]User:[/bold blue]")
                            console.print(content)
                        else:
                            console.print(f"\n[bold green]Agent:[/bold green]")
                            if env.markdown_output:
                                console.print(Markdown(content))
                            else:
                                console.print(content)

                elif command.startswith("export"):
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: .export PATH[/yellow]")
                        continue

                    path = parts[1]
                    if env.export_history(path):
                        console.print(f"[green]History exported to {path}[/green]")

                elif command == "clear":
                    env.clear_history()

                else:
                    console.print(f"[yellow]Unknown command: {command}[/yellow]")
                    console.print("[yellow]Type .help for available commands[/yellow]")

            else:
                # Process normal user input
                with console.status(
                    "[bold green]Agent is thinking...[/bold green]", spinner="dots"
                ):
                    response = await env.run_prompt(user_input)

                # Display the response
                env.display_response(response)

    except (KeyboardInterrupt, EOFError):
        console.print("\n\n[italic]REPL session ended.[/italic]")
    except Exception as e:
        console.print(f"\n[bold red]Error in REPL:[/bold red] {str(e)}")


def count_prompt_tokens(prompt: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a prompt.

    Args:
        prompt: The prompt to count tokens for
        model: The model to use for token counting

    Returns:
        The number of tokens in the prompt
    """
    try:
        return count_tokens(prompt, model)
    except Exception:
        # Fallback estimation if token counting fails
        return len(prompt.split())
