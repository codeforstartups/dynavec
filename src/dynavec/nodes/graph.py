"""Pipeline execution graph for wiring and running connected nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Node


@dataclass
class Edge:
    """Directed connection passing outputs from source node to target node."""

    source_node: str
    target_node: str
    source_port: str = "output"
    target_port: str = "input"
    mapping: dict[str, str] = field(default_factory=dict)


class SimpleGraph:
    """Directed execution pipeline to compose and run node graphs."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._execution_order: list[str] = []

    def add_node(self, node: Node) -> SimpleGraph:
        """Add a node instance to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._execution_order:
            self._execution_order.append(node.id)
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        source_port: str = "output",
        target_port: str = "input",
        mapping: dict[str, str] | None = None,
    ) -> SimpleGraph:
        """Connect source node output to target node input."""
        if source_id not in self.nodes:
            raise ValueError(f"Source node {source_id!r} not in graph.")
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id!r} not in graph.")

        edge = Edge(
            source_node=source_id,
            target_node=target_id,
            source_port=source_port,
            target_port=target_port,
            mapping=dict(mapping or {}),
        )
        self.edges.append(edge)
        return self

    def run(self, initial_inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the graph sequentially, threading state along connected edges."""
        node_outputs: dict[str, dict[str, Any]] = {}
        current_state: dict[str, Any] = dict(initial_inputs)

        for node_id in self._execution_order:
            node = self.nodes[node_id]

            # Collect inputs for this node from incoming edges or current state
            node_inputs: dict[str, Any] = dict(current_state)

            incoming_edges = [e for e in self.edges if e.target_node == node_id]
            for edge in incoming_edges:
                prev_outputs = node_outputs.get(edge.source_node, {})
                if edge.mapping:
                    for src_k, tgt_k in edge.mapping.items():
                        if src_k in prev_outputs:
                            node_inputs[tgt_k] = prev_outputs[src_k]
                else:
                    if edge.source_port in prev_outputs:
                        node_inputs[edge.target_port] = prev_outputs[edge.source_port]
                    else:
                        node_inputs.update(prev_outputs)

            outputs = node.execute(node_inputs)
            node_outputs[node_id] = outputs
            current_state.update(outputs)

        return current_state
