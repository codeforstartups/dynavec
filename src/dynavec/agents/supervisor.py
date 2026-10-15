"""Supervisor primitives for multi-agent workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Union

from .base import AgentResult
from .handoff import HandoffRequest
from .state import AgentState


class AgentLike(Protocol):
    """Interface required by a supervisor-managed agent."""

    def run(
        self,
        goal: str,
        *,
        state: AgentState,
    ) -> AgentResult:
        """Execute an agent using shared workflow state."""
        ...


Router = Callable[[str, AgentState], Union[str, list[str]]]


class Supervisor:
    """Delegate a goal to specialists using shared workflow state."""

    def __init__(
        self,
        agents: list[AgentLike] | dict[str, AgentLike],
        *,
        router: Router | None = None,
    ) -> None:
        self.agents: dict[str, AgentLike] | list[AgentLike]
        if isinstance(agents, dict):
            self.agents = dict(agents)
        else:
            self.agents = list(agents)

        self.router = router

        if router is not None and not isinstance(self.agents, dict):
            raise TypeError("A router requires agents to be provided as a named dictionary.")

        if isinstance(self.agents, dict) and not self.agents:
            raise ValueError("At least one agent must be registered.")
        if isinstance(self.agents, list) and not self.agents:
            raise ValueError("At least one agent must be registered.")

    def run(self, goal: str, *, state: AgentState) -> list[AgentResult]:
        """Run all agents sequentially or route to one selected specialist."""
        if self.router is not None:
            selected = self.router(goal, state)
            agent_names = [selected] if isinstance(selected, str) else selected

            if not agent_names:
                raise ValueError("Router must select at least one agent.")

            unknown = [name for name in agent_names if name not in self.agents]
            if unknown:
                raise ValueError(
                    f"Router selected unknown agent(s) {unknown!r}. "
                    f"Registered agents: {list(self.agents)}"
                )

        if self.router is not None:
            if not isinstance(self.agents, dict):
                raise TypeError(
                    "A router requires agents to be provided as a named dictionary."
                )

            selected = self.router(goal, state)
            agent_names = [selected] if isinstance(selected, str) else selected

            if not agent_names:
                raise ValueError("Router must select at least one agent.")

            unknown = [name for name in agent_names if name not in self.agents]
            if unknown:
                raise ValueError(
                    f"Router selected unknown agent(s) {unknown!r}. "
                    f"Registered agents: {list(self.agents)}"
                )

            return [
                self.agents[name].run(goal, state=state)
                for name in agent_names
            ]

        agents = self.agents.values() if isinstance(self.agents, dict) else self.agents
        return [agent.run(goal, state=state) for agent in agents]

    def run_and_aggregate(
        self,
        goal: str,
        *,
        state: AgentState,
    ) -> AgentResult:
        """Run the selected or registered agents and combine their outputs."""
        results = self.run(goal, state=state)
        combined_output = "\n\n".join(result.output for result in results if result.output.strip())

        return AgentResult(
            output=combined_output,
            steps=[step for result in results for step in result.steps],
            finished=all(result.finished for result in results),
            termination_reason=(
                "completed" if all(result.finished for result in results) else "incomplete"
            ),
            total_steps=sum(result.total_steps for result in results),
            tool_calls_count=sum(result.tool_calls_count for result in results),
        )

    def handoff(
        self,
        request: HandoffRequest,
        *,
        state: AgentState,
    ) -> AgentResult:
        """Transfer execution to the named specialist with shared state."""
        if not isinstance(self.agents, dict):
            raise TypeError("Handoffs require agents to be registered by name.")

        agent = self.agents.get(request.target)
        if agent is None:
            raise ValueError(
                f"Unknown handoff target {request.target!r}. "
                f"Registered agents: {list(self.agents)}"
            )

        return agent.run(request.goal, state=state)
