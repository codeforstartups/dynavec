"""Base abstractions and schemas for the dynaflow node system."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

PortType = Literal["string", "number", "boolean", "array", "object", "any"]
NodeStatus = Literal["pending", "running", "retrying", "succeeded", "failed", "skipped"]


class PortCompatibilityError(TypeError, ValueError):
    """Raised when connecting incompatible port types in a dynaflow graph."""


def is_port_type_compatible(source_type: PortType, target_type: PortType) -> bool:
    """Evaluate whether source_type can be assigned or connected to target_type.

    The 'any' port type matches any other type symmetrically.
    Exact matches (e.g. string -> string, number -> number) are compatible.
    In agent/LLM pipelines, string and object are compatible to allow JSON-encoded
    strings to flow into object ports (and vice-versa).
    """
    if source_type == "any" or target_type == "any":
        return True
    if source_type == target_type:
        return True
    if (source_type == "string" and target_type == "object") or (
        source_type == "object" and target_type == "string"
    ):
        return True
    return False


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodePort:
        """Deserialize port definition from a dictionary."""
        return cls(
            name=data["name"],
            port_type=data.get("port_type", "any"),
            description=data.get("description", ""),
            required=data.get("required", True),
            default=data.get("default", None),
        )


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

    def get_input_port(self, name: str) -> NodePort | None:
        """Retrieve an input port definition by name if declared."""
        for p in self.inputs:
            if p.name == name:
                return p
        return None

    def get_output_port(self, name: str) -> NodePort | None:
        """Retrieve an output port definition by name if declared."""
        for p in self.outputs:
            if p.name == name:
                return p
        return None

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeSchema:
        """Deserialize node schema from a dictionary."""
        inputs = [
            NodePort.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("inputs", [])
        ]
        outputs = [
            NodePort.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("outputs", [])
        ]
        return cls(
            node_type=data["node_type"],
            label=data.get("label", data["node_type"]),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            inputs=inputs,
            outputs=outputs,
            config_schema=dict(data.get("config_schema", {})),
        )

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Instantiate node from a dictionary representation."""
        node_id = data.get("id")
        node_data = data.get("data", {})
        config = node_data.get("config", {})
        retry_policy_data = node_data.get("retry_policy")
        retry_policy = (
            RetryPolicy.from_dict(retry_policy_data)
            if retry_policy_data
            else None
        )
        node = cls(node_id=node_id, config=config, retry_policy=retry_policy)
        if "status" in node_data:
            node.status = node_data["status"]
        return node
