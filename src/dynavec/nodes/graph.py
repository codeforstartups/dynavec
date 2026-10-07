"""Pipeline execution graph for wiring and running connected nodes with retries and resumability."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from .base import Node, NodeStatus, RetryPolicy


@dataclass
class Edge:
    """Directed connection passing outputs from source node to target node."""

    source_node: str
    target_node: str
    source_port: str = "output"
    target_port: str = "input"
    mapping: dict[str, str] = field(default_factory=dict)
    is_error_edge: bool = False


@dataclass
class GraphRunState:
    """State checkpoint representing the progress and status of a graph run."""

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    initial_inputs: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_statuses: dict[str, NodeStatus] = field(default_factory=dict)
    node_errors: dict[str, str] = field(default_factory=dict)
    completed: bool = False
    failed_node: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary for persistent checkpoint storage."""
        return {
            "run_id": self.run_id,
            "initial_inputs": dict(self.initial_inputs),
            "current_state": dict(self.current_state),
            "node_outputs": dict(self.node_outputs),
            "node_statuses": dict(self.node_statuses),
            "node_errors": dict(self.node_errors),
            "completed": self.completed,
            "failed_node": self.failed_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphRunState:
        """Reconstruct state checkpoint from dictionary."""
        return cls(
            run_id=data.get("run_id", f"run_{uuid.uuid4().hex[:8]}"),
            initial_inputs=dict(data.get("initial_inputs", {})),
            current_state=dict(data.get("current_state", {})),
            node_outputs=dict(data.get("node_outputs", {})),
            node_statuses=dict(data.get("node_statuses", {})),
            node_errors=dict(data.get("node_errors", {})),
            completed=data.get("completed", False),
            failed_node=data.get("failed_node"),
        )


class SimpleGraph:
    """Directed execution pipeline to compose and run resilient node graphs."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._execution_order: list[str] = []
        self.last_state: GraphRunState | None = None

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
        is_error_edge: bool = False,
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
            is_error_edge=is_error_edge,
        )
        self.edges.append(edge)
        return self

    def add_error_edge(
        self,
        source_id: str,
        target_id: str,
        source_port: str = "error",
        target_port: str = "error",
        mapping: dict[str, str] | None = None,
    ) -> SimpleGraph:
        """Add a fallback error edge triggered when source node fails execution."""
        return self.add_edge(
            source_id=source_id,
            target_id=target_id,
            source_port=source_port,
            target_port=target_port,
            mapping=mapping,
            is_error_edge=True,
        )

    def _execute_node_with_retry(
        self, node: Node, node_inputs: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        """Execute a node applying its configured RetryPolicy if present."""
        policy: RetryPolicy | None = getattr(node, "retry_policy", None)
        max_attempts = policy.max_attempts if policy is not None else 1
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            node.attempts = attempt
            try:
                if attempt > 1:
                    node.status = "retrying"
                else:
                    node.status = "running"

                outputs = node.execute(node_inputs)
                node.status = "succeeded"
                node.last_error = None
                return outputs, None
            except Exception as exc:  # noqa: BLE001
                node.last_error = str(exc)
                if policy is not None and attempt < max_attempts and policy.should_retry(exc):
                    delay = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
                    if policy.jitter:
                        delay = random.uniform(0, delay)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                node.status = "failed"
                return None, exc

        node.status = "failed"
        return None, RuntimeError(f"Node {node.id!r} exhausted {max_attempts} attempts.")

    @overload
    def run(
        self,
        initial_inputs: dict[str, Any],
        state: GraphRunState | dict[str, Any] | None = None,
        return_state: Literal[False] = False,
        raise_on_failure: bool = True,
    ) -> dict[str, Any]:
        ...

    @overload
    def run(
        self,
        initial_inputs: dict[str, Any],
        state: GraphRunState | dict[str, Any] | None = None,
        return_state: Literal[True] = ...,
        raise_on_failure: bool = True,
    ) -> GraphRunState:
        ...

    @overload
    def run(
        self,
        initial_inputs: dict[str, Any],
        state: GraphRunState | dict[str, Any] | None = None,
        return_state: bool = False,
        raise_on_failure: bool = True,
    ) -> dict[str, Any] | GraphRunState:
        ...

    def run(
        self,
        initial_inputs: dict[str, Any],
        state: GraphRunState | dict[str, Any] | None = None,
        return_state: bool = False,
        raise_on_failure: bool = True,
    ) -> dict[str, Any] | GraphRunState:
        """Execute the graph with retries, error edges, and state tracking."""
        if isinstance(state, dict):
            run_state = GraphRunState.from_dict(state)
        elif isinstance(state, GraphRunState):
            run_state = state
        else:
            run_state = GraphRunState(
                initial_inputs=dict(initial_inputs),
                current_state=dict(initial_inputs),
            )

        current_state = run_state.current_state
        node_outputs = run_state.node_outputs
        node_statuses = run_state.node_statuses
        node_errors = run_state.node_errors

        for node_id in self._execution_order:
            node = self.nodes[node_id]

            # If node already succeeded in previous checkpoint, reuse output and skip
            if node_statuses.get(node_id) == "succeeded" and node_id in node_outputs:
                node.status = "succeeded"
                node.attempts = 1
                continue

            # Check if previous normal dependencies failed
            incoming_standard_edges = [
                e for e in self.edges if e.target_node == node_id and not e.is_error_edge
            ]
            incoming_error_edges = [
                e for e in self.edges if e.target_node == node_id and e.is_error_edge
            ]

            # Determine if this node is triggered via an error edge or standard execution
            is_error_target = False
            error_payload: dict[str, Any] = {}
            for e in incoming_error_edges:
                if node_statuses.get(e.source_node) == "failed":
                    is_error_target = True
                    error_payload = {
                        "error": node_errors.get(e.source_node, "Unknown error"),
                        "failed_node": e.source_node,
                    }
                    break

            # Collect inputs for this node
            node_inputs: dict[str, Any] = dict(current_state)

            if is_error_target:
                node_inputs.update(error_payload)
            else:
                for edge in incoming_standard_edges:
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

            # Execute node with retry logic
            outputs, exc = self._execute_node_with_retry(node, node_inputs)
            node_statuses[node_id] = node.status

            if exc is not None:
                node_errors[node_id] = str(exc)
                run_state.failed_node = node_id

                # Check if there is an outgoing error edge to recover from this failure
                outgoing_error_edges = [
                    e for e in self.edges if e.source_node == node_id and e.is_error_edge
                ]
                if not outgoing_error_edges:
                    run_state.completed = False
                    self.last_state = run_state
                    if raise_on_failure:
                        raise exc
                    return run_state if return_state else current_state
            else:
                if outputs is not None:
                    node_outputs[node_id] = outputs
                    current_state.update(outputs)

        run_state.completed = all(s == "succeeded" for s in node_statuses.values())
        run_state.failed_node = None if run_state.completed else run_state.failed_node
        self.last_state = run_state

        if return_state:
            return run_state
        return current_state

    def resume(
        self,
        state: GraphRunState | dict[str, Any],
        override_inputs: dict[str, Any] | None = None,
        raise_on_failure: bool = True,
    ) -> GraphRunState:
        """Resume a partially-completed or failed graph run from a checkpoint."""
        if isinstance(state, dict):
            run_state = GraphRunState.from_dict(state)
        else:
            run_state = state

        if override_inputs:
            run_state.current_state.update(override_inputs)

        # Clear failure marker on resumed run
        run_state.failed_node = None
        result = self.run(
            initial_inputs=run_state.initial_inputs,
            state=run_state,
            return_state=True,
            raise_on_failure=raise_on_failure,
        )
        assert isinstance(result, GraphRunState)
        return result
