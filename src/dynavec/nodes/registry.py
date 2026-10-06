"""Node registry for node discovery and canvas palette serialization."""

from __future__ import annotations

from typing import Any

from .base import Node, NodeSchema


class NodeRegistry:
    """Registry maintaining available node types and their schemas."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Node]] = {}

    def register(self, node_cls: type[Node]) -> type[Node]:
        """Register a node class under its node_type."""
        if not hasattr(node_cls, "node_type"):
            raise ValueError(f"Class {node_cls.__name__} must define a 'node_type' attribute.")
        self._registry[node_cls.node_type] = node_cls
        return node_cls

    def get(self, node_type: str) -> type[Node] | None:
        """Lookup a registered node class by type."""
        return self._registry.get(node_type)

    def list_types(self) -> list[str]:
        """Return all registered node type identifiers."""
        return sorted(self._registry.keys())

    def get_schemas(self) -> list[NodeSchema]:
        """Return schemas for all registered nodes."""
        schemas: list[NodeSchema] = []
        for n_type in self.list_types():
            cls = self._registry[n_type]
            if hasattr(cls, "schema"):
                schemas.append(cls.schema)
        return schemas

    def export_canvas_palette(self) -> list[dict[str, Any]]:
        """Export ReactFlow-compatible component palette for canvas composition."""
        return [s.to_reactflow_dict() for s in self.get_schemas()]


node_registry = NodeRegistry()
