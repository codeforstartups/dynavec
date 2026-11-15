"""Node registry for node discovery and canvas palette serialization."""

from __future__ import annotations

from typing import Any

from .base import Node, NodeSchema


class NodeRegistry:
    """Registry maintaining available node types and their schemas."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Node]] = {}

    def register(self, node_cls: type[Node] | None = None) -> Any:
        """Register a node class under its node_type.

        Can be used as a function call `node_registry.register(MyNode)` or
        as a decorator `@node_registry.register` or `@node_registry.register()`.
        """
        def _do_register(cls: type[Node]) -> type[Node]:
            if not hasattr(cls, "node_type"):
                raise ValueError(f"Class {cls.__name__} must define a 'node_type' attribute.")
            self._registry[cls.node_type] = cls
            return cls

        if node_cls is not None:
            return _do_register(node_cls)
        return _do_register

    def unregister(self, node_type: str) -> bool:
        """Unregister a node type by its identifier.

        Returns True if the node type was removed, False if it was not registered.
        """
        if node_type in self._registry:
            del self._registry[node_type]
            return True
        return False

    def has(self, node_type: str) -> bool:
        """Check if a node type is currently registered."""
        return node_type in self._registry

    def get(self, node_type: str) -> type[Node] | None:
        """Lookup a registered node class by type."""
        return self._registry.get(node_type)

    def get_schema(self, node_type: str) -> NodeSchema | None:
        """Retrieve the schema for a registered node type."""
        cls = self.get(node_type)
        if cls is not None and hasattr(cls, "schema"):
            return cls.schema
        return None

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

    def clear(self) -> None:
        """Remove all registered node types."""
        self._registry.clear()


node_registry = NodeRegistry()
