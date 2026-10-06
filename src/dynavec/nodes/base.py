"""Base abstractions and schemas for the dynaflow node system."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

PortType = Literal["string", "number", "boolean", "array", "object", "any"]


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
    ) -> None:
        self.id = node_id or f"{self.node_type}_{uuid.uuid4().hex[:8]}"
        self.config = dict(config or {})

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the node logic given an inputs dictionary, returning outputs."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to a ReactFlow-compatible node object."""
        return {
            "id": self.id,
            "type": self.node_type,
            "data": {
                "label": self.schema.label,
                "config": self.config,
                "category": self.schema.category,
            },
            "position": {"x": 0, "y": 0},
        }
