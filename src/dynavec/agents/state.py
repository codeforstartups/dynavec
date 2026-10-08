"""Shared state primitives for multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Shared state passed between agents in a workflow."""

    data: dict[str, Any] = field(default_factory=dict)
