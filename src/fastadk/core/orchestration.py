"""
Multi-agent orchestration for FastADK.

This module provides utilities for coordinating multiple agents, enabling them
to work together on complex tasks and communicate with each other.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from .agent import BaseAgent
from .exceptions import AgentError, OrchestrationError

logger = logging.getLogger("fastadk.orchestration")


class OrchestrationStrategy(str, Enum):
    """Strategies for orchestrating multiple agents."""

    SEQUENTIAL = "sequential"  # Agents run one after another
    BROADCAST = "broadcast"  # Same input sent to all agents
    DYNAMIC = "dynamic"  # Agents determine which other agents to call


class OrchestrationResult:
    """Results from an orchestration run."""

    def __init__(self) -> None:
        """Initialize the orchestration result."""
        self.outputs: Dict[str, Any] = {}
        self.intermediate_steps: List[Dict[str, Any]] = []
        self.errors: Dict[str, Exception] = {}
        self._final_output: Optional[str] = None

    @property
    def final_output(self) -> str:
        """Get the final output from the orchestration."""
        if self._final_output is not None:
            return self._final_output

        # If no final output was explicitly set, return the last agent's output
        if self.outputs:
            last_agent = list(self.outputs.keys())[-1]
            return str(self.outputs[last_agent])

        return "No output produced by orchestration."

    @final_output.setter
    def final_output(self, value: str) -> None:
        """Set the final output from the orchestration."""
        self._final_output = value

    def add_output(self, agent_name: str, output: Any) -> None:
        """
        Add an output from an agent.

        Args:
            agent_name: The name of the agent
            output: The output from the agent
        """
        self.outputs[agent_name] = output

    def add_error(self, agent_name: str, error: Exception) -> None:
        """
        Add an error from an agent.

        Args:
            agent_name: The name of the agent
            error: The error that occurred
        """
        self.errors[agent_name] = error

    def add_step(self, step_data: Dict[str, Any]) -> None:
        """
        Add an intermediate step to the result.

        Args:
            step_data: Data about the step
        """
        self.intermediate_steps.append(step_data)

    def has_errors(self) -> bool:
        """
        Check if there were any errors during orchestration.

        Returns:
            True if there were errors, False otherwise
        """
        return bool(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "outputs": self.outputs,
            "intermediate_steps": self.intermediate_steps,
            "errors": {k: str(v) for k, v in self.errors.items()},
            "final_output": self.final_output,
            "has_errors": self.has_errors(),
        }


class Orchestrator:
    """
    Orchestrates multiple agents to work together on complex tasks.

    The Orchestrator class provides utilities for coordinating multiple agents,
    enabling them to communicate with each other and work together to solve
    complex problems.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        strategy: Union[str, OrchestrationStrategy] = OrchestrationStrategy.SEQUENTIAL,
    ) -> None:
        """
        Initialize the orchestrator with a list of agents.

        Args:
            agents: List of agent instances to orchestrate
            strategy: Orchestration strategy to use
        """
        self.agents = agents
        self.agent_map: Dict[str, BaseAgent] = {}

        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            try:
                self.strategy = OrchestrationStrategy(strategy.lower())
            except ValueError as e:
                raise ValueError(
                    f"Invalid orchestration strategy: {strategy}. "
                    f"Valid options: {', '.join([s.value for s in OrchestrationStrategy])}"
                ) from e
        else:
            self.strategy = strategy

        # Map agents by name for easy access
        for agent in agents:
            agent_name = agent.__class__.__name__
            if agent_name in self.agent_map:
                # If duplicate name, append an index
                count = 1
                while f"{agent_name}_{count}" in self.agent_map:
                    count += 1
                agent_name = f"{agent_name}_{count}"

            self.agent_map[agent_name] = agent

        logger.info(
            "Initialized orchestrator with %d agents using %s strategy",
            len(agents),
            self.strategy.value,
        )

    async def run(
        self,
        input_text: str,
        agent_order: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Run the orchestration with the given input.

        Args:
            input_text: The initial input text
            agent_order: Optional list of agent names specifying the execution order
            context: Optional context data to pass between agents

        Returns:
            OrchestrationResult containing outputs from all agents

        Raises:
            OrchestrationError: If there's an error during orchestration
        """
        result = OrchestrationResult()
        context = context or {}

        try:
            if self.strategy == OrchestrationStrategy.SEQUENTIAL:
                await self._run_sequential(input_text, result, agent_order, context)
            elif self.strategy == OrchestrationStrategy.BROADCAST:
                await self._run_broadcast(input_text, result, context)
            elif self.strategy == OrchestrationStrategy.DYNAMIC:
                await self._run_dynamic(input_text, result, context)
            else:
                raise OrchestrationError(
                    f"Unsupported orchestration strategy: {self.strategy}"
                )

            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error during orchestration: %s", str(e), exc_info=True)
            if not isinstance(e, OrchestrationError):
                raise OrchestrationError(f"Orchestration failed: {str(e)}") from e
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during orchestration: %s", str(e), exc_info=True
            )
            raise OrchestrationError(
                f"Orchestration failed with unexpected error: {str(e)}"
            ) from e

    async def _run_sequential(
        self,
        input_text: str,
        result: OrchestrationResult,
        agent_order: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run agents sequentially, with each agent receiving the output of the previous one.

        Args:
            input_text: The initial input text
            result: The orchestration result to update
            agent_order: Optional list of agent names specifying the execution order
            context: Context data to pass between agents
        """
        current_input = input_text
        context = context or {}

        # Determine agent execution order
        agents_to_run = []
        if agent_order:
            # Use specified order
            for agent_name in agent_order:
                if agent_name not in self.agent_map:
                    raise OrchestrationError(
                        f"Agent '{agent_name}' not found in orchestrator"
                    )
                agents_to_run.append((agent_name, self.agent_map[agent_name]))
        else:
            # Use order from initialization
            agents_to_run = list(self.agent_map.items())

        # Run agents in sequence
        for agent_name, agent in agents_to_run:
            try:
                logger.info(
                    "Running agent '%s' with input: %s...",
                    agent_name,
                    current_input[:100],
                )

                # Run the agent
                output = await agent.run(current_input)

                # Store the result
                result.add_output(agent_name, output)
                result.add_step(
                    {
                        "agent": agent_name,
                        "input": current_input,
                        "output": output,
                    }
                )

                # Update input for next agent
                current_input = output

                logger.info(
                    "Agent '%s' produced output: %s...", agent_name, output[:100]
                )

            except (ValueError, KeyError, AttributeError) as e:
                logger.error(
                    "Error running agent '%s': %s", agent_name, str(e), exc_info=True
                )
                result.add_error(agent_name, e)
                result.add_step(
                    {
                        "agent": agent_name,
                        "input": current_input,
                        "error": str(e),
                    }
                )

                # If configured to stop on error, break the loop
                if context.get("stop_on_error", True):
                    logger.info("Stopping orchestration due to error")
                    break

        # Set the final output to the last successful agent's output
        result.final_output = current_input

    async def _run_broadcast(
        self,
        input_text: str,
        result: OrchestrationResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send the same input to all agents and collect their responses.

        Args:
            input_text: The input text to send to all agents
            result: The orchestration result to update
            context: Context data to pass to agents
        """
        context = context or {}
        tasks = []

        # Create tasks for all agents
        for agent_name, agent in self.agent_map.items():
            tasks.append(self._run_agent(agent_name, agent, input_text, result))

        # Run all agents concurrently
        await asyncio.gather(*tasks)

        # Combine outputs if requested
        if context.get("combine_outputs", False):
            combined = "\n\n".join(
                [
                    f"### {agent_name}:\n{output}"
                    for agent_name, output in result.outputs.items()
                ]
            )
            result.final_output = combined

    async def _run_dynamic(
        self,
        input_text: str,
        result: OrchestrationResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run agents dynamically, where agents can call other agents.

        Args:
            input_text: The initial input text
            result: The orchestration result to update
            context: Context data to pass to agents
        """
        context = context or {}
        visited_agents: Set[str] = set()

        # Start with the first agent or a specified starting agent
        start_agent = context.get("start_agent", list(self.agent_map.keys())[0])
        if start_agent not in self.agent_map:
            raise OrchestrationError(f"Starting agent '{start_agent}' not found")

        # Add methods to each agent to call other agents
        self._setup_agent_calling(result, visited_agents)

        try:
            # Run the starting agent
            agent = self.agent_map[start_agent]
            output = await agent.run(input_text)

            # Store the result
            result.add_output(start_agent, output)
            result.add_step(
                {
                    "agent": start_agent,
                    "input": input_text,
                    "output": output,
                }
            )

            # Set the final output to the last agent's output
            result.final_output = output

            logger.info(
                "Dynamic orchestration completed, visited %d agents",
                len(visited_agents),
            )

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error in dynamic orchestration: %s", str(e), exc_info=True)
            result.add_error("orchestration", e)

    def _setup_agent_calling(
        self,
        result: OrchestrationResult,
        visited_agents: Set[str],
    ) -> None:
        """
        Set up agents to be able to call each other.

        Args:
            result: The orchestration result to update
            visited_agents: Set to track which agents have been visited
        """

        # Create a call_agent method that will be added to each agent
        async def call_agent(
            self_agent: BaseAgent, agent_name: str, input_text: str
        ) -> str:
            """
            Call another agent from within an agent.

            Args:
                agent_name: Name of the agent to call
                input_text: Input to send to the agent

            Returns:
                The output from the called agent
            """
            if agent_name not in orchestrator.agent_map:
                raise AgentError(f"Agent '{agent_name}' not found")

            # Prevent circular calls and infinite loops
            if agent_name in visited_agents:
                logger.warning(
                    "Agent '%s' has already been called in this chain",
                    agent_name,
                )
                return f"[Agent '{agent_name}' has already been called in this chain]"

            visited_agents.add(agent_name)
            called_agent = orchestrator.agent_map[agent_name]

            try:
                logger.info(
                    "Agent '%s' calling agent '%s'",
                    self_agent.__class__.__name__,
                    agent_name,
                )
                output = await called_agent.run(input_text)

                # Store the result
                result.add_output(agent_name, output)
                result.add_step(
                    {
                        "agent": agent_name,
                        "caller": self_agent.__class__.__name__,
                        "input": input_text,
                        "output": output,
                    }
                )

                return output

            except (ValueError, KeyError, AttributeError) as e:
                logger.error(
                    "Error calling agent '%s': %s", agent_name, str(e), exc_info=True
                )
                result.add_error(agent_name, e)
                return f"[Error calling agent '{agent_name}': {str(e)}]"

        # Store reference to self for the injected method
        orchestrator = self

        # Add the call_agent method to each agent
        for agent in self.agents:
            # Add the method to the agent instance
            # We use type: ignore to suppress mypy error since BaseAgent doesn't have call_agent attribute
            agent.call_agent = call_agent.__get__(agent, agent.__class__)  # type: ignore

    async def _run_agent(
        self,
        agent_name: str,
        agent: BaseAgent,
        input_text: str,
        result: OrchestrationResult,
    ) -> None:
        """
        Run a single agent and add its result to the orchestration result.

        Args:
            agent_name: Name of the agent
            agent: The agent instance
            input_text: Input text for the agent
            result: Orchestration result to update
        """
        try:
            truncated_input = input_text[:100]
            logger.info(
                "Running agent '%s' with input: %s...", agent_name, truncated_input
            )

            # Run the agent
            output = await agent.run(input_text)

            # Store the result
            result.add_output(agent_name, output)
            result.add_step(
                {
                    "agent": agent_name,
                    "input": input_text,
                    "output": output,
                }
            )

            logger.info("Agent '%s' produced output: %s...", agent_name, output[:100])

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(
                "Error running agent '%s': %s", agent_name, str(e), exc_info=True
            )
            result.add_error(agent_name, e)
            result.add_step(
                {
                    "agent": agent_name,
                    "input": input_text,
                    "error": str(e),
                }
            )
