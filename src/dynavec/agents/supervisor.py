"""Supervisor primitives for multi-agent workflows."""

from __future__ import annotations

from typing import Protocol

from .base import AgentResult
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


class Supervisor:
    """Delegate a goal to multiple agents using shared workflow state."""

    def __init__(self, agents: list[AgentLike]) -> None:
        self.agents = agents

    def run(self, goal: str, *, state: AgentState) -> list[AgentResult]:
        """Run each specialist with the same shared state."""
        return [agent.run(goal, state=state) for agent in self.agents]
