"""Handoff primitives for multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import AgentResult
from .state import AgentState

if TYPE_CHECKING:
    from .supervisor import AgentLike


@dataclass(frozen=True)
class HandoffRequest:
    """Request graph execution to continue at a named specialist."""

    target: str
    goal: str


class Handoff:
    """Pass workflow state through or request transfer to a named specialist."""

    def __call__(
        self,
        target: str | AgentState,
        goal: str | None = None,
    ) -> HandoffRequest | AgentState:
        """Preserve legacy state passthrough and support named handoff requests."""
        if isinstance(target, AgentState):
            if goal is not None:
                raise TypeError("A goal cannot be supplied when passing through state.")
            return target

        if goal is None:
            raise TypeError("A goal is required for a named handoff.")

        if not target.strip():
            raise ValueError("Handoff target must not be empty.")
        if not goal.strip():
            raise ValueError("Handoff goal must not be empty.")

        return HandoffRequest(target=target, goal=goal)

    def transfer(
        self,
        target: AgentLike,
        goal: str,
        *,
        state: AgentState,
    ) -> AgentResult:
        """Invoke a target specialist with the existing workflow state."""
        return target.run(goal, state=state)
