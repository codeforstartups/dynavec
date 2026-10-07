"""Base abstractions and schemas for the dynaflow node system."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

PortType = Literal["string", "number", "boolean", "array", "object", "any"]
NodeStatus = Literal["pending", "running", "retrying", "succeeded", "failed", "skipped"]


@dataclass
class RetryPolicy:
    """Configurable retry policy for node execution with exponential backoff and jitter."""

    max_attempts: int = 3
    base_delay: float = 0.05
    max_delay: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] | None = None

    def should_retry(self, exc: Exception) -> bool:
        """Evaluate if an exception is eligible for retry under this policy."""
        if self.retry_on is None:
            return True
        if isinstance(self.retry_on, tuple):
            return isinstance(exc, self.retry_on)
        if callable(self.retry_on):
            return bool(self.retry_on(exc))
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize retry policy to dictionary."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryPolicy:
        """Deserialize retry policy from dictionary."""
        return cls(
            max_attempts=data.get("max_attempts", 3),
            base_delay=data.get("base_delay", 0.05),
            max_delay=data.get("max_delay", 2.0),
            jitter=data.get("jitter", True),
        )


@dataclass
class NodePort:
    """A typed input or output connection port on a node."""

    name: str
    port_type: PortType = "any"
    description: str = ""
    required: bool = True
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert port definition to a dictionary."""
        return {
            "name": self.name,
            "port_type": self.port_type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }


@dataclass
class NodeSchema:
    """Metadata describing a node type's ports, config, and canvas properties."""

    node_type: str
    label: str
    description: str
    category: str = "general"
    inputs: list[NodePort] = field(default_factory=list)
    outputs: list[NodePort] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "node_type": self.node_type,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "config_schema": self.config_schema,
        }

    def to_reactflow_dict(self) -> dict[str, Any]:
        """Export ReactFlow-compatible component metadata for visual canvas builders."""
        return {
            "type": self.node_type,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "inputPorts": [p.name for p in self.inputs],
            "outputPorts": [p.name for p in self.outputs],
            "config": self.config_schema,
        }


class Node(ABC):
    """Abstract base class for all execution nodes in dynaflow."""

    node_type: str
    schema: NodeSchema

    def __init__(
        self,
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.id = node_id or f"{self.node_type}_{uuid.uuid4().hex[:8]}"
        self.config = dict(config or {})
        self.retry_policy = retry_policy
        self.status: NodeStatus = "pending"
        self.attempts: int = 0
        self.last_error: str | None = None

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the node logic given an inputs dictionary, returning outputs."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to a ReactFlow-compatible node object."""
        data: dict[str, Any] = {
            "label": self.schema.label,
            "config": self.config,
            "category": self.schema.category,
            "status": self.status,
        }
        if self.retry_policy is not None:
            data["retry_policy"] = self.retry_policy.to_dict()

        return {
            "id": self.id,
            "type": self.node_type,
            "data": data,
            "position": {"x": 0, "y": 0},
        }
