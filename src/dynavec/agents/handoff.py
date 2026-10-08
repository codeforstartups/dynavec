"""Handoff primitives for multi-agent workflows."""

from __future__ import annotations

from .state import AgentState


class Handoff:
    """Pass shared agent state from one workflow step to another."""

    def __call__(self, state: AgentState) -> AgentState:
        """Return the same state so downstream agents share workflow context."""
        return state
