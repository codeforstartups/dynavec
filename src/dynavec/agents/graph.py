"""Agent graph orchestration engine for dynaflow.

This module provides the graph execution runtime: shared state with reducers,
nodes, conditional routing edges, parallel fan-out and join (barrier) execution,
and bounded loops with termination guards and cycle validation.
"""

from __future__ import annotations

import ast
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GraphError(Exception):
    """Base exception for agent graph execution and validation errors."""


class GraphValidationError(GraphError):
    """Raised when a graph structure is invalid or malformed."""


class GraphCycleError(GraphError):
    """Raised when a graph contains an un-guarded or infinite cycle."""


class MaxIterationsExceededError(GraphError):
    """Raised when a loop execution exceeds the maximum allowed iterations."""


# ---------------------------------------------------------------------------
# GraphState & Reducers
# ---------------------------------------------------------------------------

ReducerFn = Callable[[Any, Any], Any]


def default_reducer(current: Any, update: Any) -> Any:
    """Default state reducer: replaces current value with the update."""
    return update


def append_reducer(current: Any, update: Any) -> list[Any]:
    """Reducer that appends or extends lists."""
    if current is None:
        return list(update) if isinstance(update, (list, tuple)) else [update]
    if isinstance(current, list):
        if isinstance(update, list):
            return current + update
        return [*current, update]
    return [current, update]


class GraphState(dict[str, Any]):
    """Typed shared state that flows through the agent graph.

    Supports isolated branch copies and deterministic merging using
    custom reducer functions per key.
    """

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        *,
        reducers: dict[str, ReducerFn] | None = None,
    ) -> None:
        super().__init__(initial or {})
        self._reducers: dict[str, ReducerFn] = reducers or {}

    @property
    def reducers(self) -> dict[str, ReducerFn]:
        """Mapping of field keys to their reducer functions."""
        return self._reducers

    def register_reducer(self, key: str, reducer: ReducerFn) -> GraphState:
        """Register a custom reducer for a state key."""
        self._reducers[key] = reducer
        return self

    def set(self, key: str, value: Any) -> GraphState:
        """Set a key and return self for fluent chaining."""
        self[key] = value
        return self

    def copy(self) -> GraphState:
        """Create an isolated deep-copy of the state for parallel branches."""
        new_state = GraphState(dict(self), reducers=dict(self._reducers))
        return new_state

    def merge(self, updates: dict[str, Any] | GraphState) -> None:
        """Merge a dictionary of updates using defined reducers."""
        for key, val in updates.items():
            if key in self and key in self._reducers:
                self[key] = self._reducers[key](self[key], val)
            else:
                self[key] = val


# Backwards compatibility alias
AgentState = GraphState


# ---------------------------------------------------------------------------
# BaseNode & FunctionNode
# ---------------------------------------------------------------------------


class BaseNode(ABC):
    """Abstract base class for all nodes in an agent graph."""

    node_type: str = "base"

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState | dict[str, Any] | None:
        """Execute the node logic against current state.

        Returns updated state or dict of updates, or None if in-place mutation.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize node metadata for canvas / inspection."""
        return {"name": self.name, "type": self.node_type}


class FunctionNode(BaseNode):
    """Node wrapping a callable function: ``fn(state) -> dict | None``."""

    node_type: str = "function"

    def __init__(
        self,
        fn: Callable[[GraphState], GraphState | dict[str, Any] | None],
        name: str = "",
    ) -> None:
        super().__init__(name=name or getattr(fn, "__name__", "fn"))
        self.fn = fn

    def execute(self, state: GraphState) -> GraphState | dict[str, Any] | None:
        return self.fn(state)


# ---------------------------------------------------------------------------
# Safe Expression Evaluation
# ---------------------------------------------------------------------------

_COMPARE_RE = re.compile(r"^([\w][\w.]*)\s*(==|!=|>=|<=|>|<|in|not in)\s*(.+)$")


def safe_eval_expr(expr: str, state: dict[str, Any]) -> bool:
    """Safely evaluate comparison expressions against state without eval().

    Supports:
        field == value       field != value
        field > value        field >= value
        field < value        field <= value
        field in value       field not in value
        expr and expr        expr or expr
    """
    expr = expr.strip()

    # Precedence: 'or' first
    if " or " in expr:
        left, right = expr.split(" or ", 1)
        return safe_eval_expr(left, state) or safe_eval_expr(right, state)

    # Then 'and'
    if " and " in expr:
        left, right = expr.split(" and ", 1)
        return safe_eval_expr(left, state) and safe_eval_expr(right, state)

    m = _COMPARE_RE.match(expr)
    if not m:
        raise ValueError(f"Cannot parse condition expression: {expr!r}")

    field_name, op, raw_value = m.group(1), m.group(2), m.group(3).strip()

    # Resolve dotted key access (e.g. "result.score")
    actual: Any = state
    for part in field_name.split("."):
        if isinstance(actual, dict):
            actual = actual.get(part)
        else:
            actual = getattr(actual, part, None)

    try:
        expected = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse value {raw_value!r} in expression: {exc}") from exc

    ops: dict[str, Callable[[Any, Any], bool]] = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a is not None and a > b,
        ">=": lambda a, b: a is not None and a >= b,
        "<": lambda a, b: a is not None and a < b,
        "<=": lambda a, b: a is not None and a <= b,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
    }
    return ops[op](actual, expected)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@dataclass
class Edge:
    """Directed connection between two nodes with optional gating condition.

    Condition can be:
    * None: Unconditional transition.
    * Callable[[GraphState], bool]: Predicate evaluated against state.
    * str: Safe expression string like "confidence >= 0.8".
    """

    source: str
    target: str
    condition: Callable[[GraphState], bool] | str | None = None
    label: str = ""

    def evaluate(self, state: GraphState) -> bool:
        """Evaluate if the edge condition is satisfied."""
        if self.condition is None:
            return True
        if callable(self.condition):
            return bool(self.condition(state))
        if isinstance(self.condition, str):
            return safe_eval_expr(self.condition, state)
        raise TypeError(f"Unsupported edge condition type: {type(self.condition)}")

    def is_conditional(self) -> bool:
        """True if edge is guarded by a condition."""
        return self.condition is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source, "target": self.target}
        if self.label:
            d["label"] = self.label
        if isinstance(self.condition, str):
            d["condition"] = self.condition
        elif self.condition is not None:
            d["condition"] = f"<callable:{getattr(self.condition, '__name__', 'fn')}>"
        return d


# ---------------------------------------------------------------------------
# Observability Traces & Results
# ---------------------------------------------------------------------------


@dataclass
class NodeTrace:
    """Execution telemetry for a single node."""

    node_name: str
    node_type: str
    started_at: float
    ended_at: float
    duration_ms: float
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "node_type": self.node_type,
            "duration_ms": round(self.duration_ms, 3),
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "error": self.error,
        }


@dataclass
class GraphResult:
    """Result returned upon completion of an AgentGraph run."""

    state: GraphState
    trace: list[NodeTrace] = field(default_factory=list)
    nodes_executed: int = 0
    total_ms: float = 0.0
    graph_name: str = ""
    success: bool = True
    error: str | None = None

    @property
    def output(self) -> GraphState:
        """Convenience alias for the final state."""
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "success": self.success,
            "total_ms": round(self.total_ms, 3),
            "nodes_executed": self.nodes_executed,
            "trace": [t.to_dict() for t in self.trace],
            "state": dict(self.state),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# AgentGraph Orchestration Engine
# ---------------------------------------------------------------------------


class AgentGraph:
    """Directed graph execution engine supporting conditional routing,
    parallel fan-out/join barriers, and bounded cycles with iteration guards.
    """

    def __init__(
        self,
        name: str = "dynaflow-graph",
        *,
        description: str = "",
        max_iterations: int = 25,
    ) -> None:
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self._nodes: dict[str, BaseNode] = {}
        self._edges: list[Edge] = []
        self._entry: str | None = None
        self._join_nodes: set[str] = set()

    # ---------------------------------------------------------------- Builder

    def add_node(
        self,
        name: str,
        node: BaseNode | Callable[[GraphState], GraphState | dict[str, Any] | None],
    ) -> AgentGraph:
        """Register a node in the graph."""
        if name in self._nodes:
            raise GraphValidationError(f"Duplicate node name: {name!r}")
        if isinstance(node, BaseNode):
            node.name = name
            self._nodes[name] = node
        elif callable(node):
            self._nodes[name] = FunctionNode(node, name=name)
        else:
            raise TypeError(f"Expected BaseNode or callable, got {type(node).__name__}")
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        condition: Callable[[GraphState], bool] | str | None = None,
        label: str = "",
    ) -> AgentGraph:
        """Add a directed edge between source and target, optionally gated by condition."""
        self._edges.append(
            Edge(
                source=source,
                target=target,
                condition=condition,
                label=label,
            )
        )
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[GraphState], str | list[str]] | dict[Any, str],
        *,
        branches: dict[str, str] | None = None,
    ) -> AgentGraph:
        """Add conditional routing edges from a source node using a router callable or map."""
        if callable(router):
            if branches:
                for branch_key, target in branches.items():

                    def _make_cond(key: str) -> Callable[[GraphState], bool]:
                        def _cond(s: GraphState) -> bool:
                            res = router(s)
                            if isinstance(res, list):
                                return key in res
                            return res == key

                        return _cond

                    self.add_edge(
                        source,
                        target,
                        condition=_make_cond(branch_key),
                        label=str(branch_key),
                    )
            else:
                # Direct router returning node name(s)
                # We register an edge to all potential nodes evaluated at runtime
                raise GraphValidationError(
                    "add_conditional_edges with a router function requires the 'branches' mapping."
                )
        elif isinstance(router, dict):
            for condition_expr, target in router.items():
                if isinstance(condition_expr, str):
                    self.add_edge(source, target, condition=condition_expr)
                else:
                    raise TypeError("Router mapping keys must be string condition expressions")
        return self

    def add_parallel_branches(
        self,
        source: str,
        targets: list[str],
        *,
        join_at: str | None = None,
    ) -> AgentGraph:
        """Convenience method to fan-out from source to targets and join at join_at."""
        for target in targets:
            self.add_edge(source, target)
        if join_at:
            self._join_nodes.add(join_at)
            for target in targets:
                self.add_edge(target, join_at)
        return self

    def register_join_barrier(self, node_name: str) -> AgentGraph:
        """Designate a node as a join barrier that must wait for all inbound incoming edges."""
        self._join_nodes.add(node_name)
        return self

    def set_entry(self, name: str) -> AgentGraph:
        """Designate the entry point node of the graph."""
        if name not in self._nodes:
            raise GraphValidationError(f"Entry node {name!r} not found in registered nodes")
        self._entry = name
        return self

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    # ------------------------------------------------------------- Validation

    def validate(self) -> list[str]:
        """Validate graph topology, catching disconnected nodes and infinite malformed cycles."""
        errors: list[str] = []

        if self._entry is None:
            errors.append("No entry node set (call set_entry())")
        elif self._entry not in self._nodes:
            errors.append(f"Entry node {self._entry!r} not found")

        for edge in self._edges:
            if edge.source not in self._nodes:
                errors.append(f"Edge source {edge.source!r} not found")
            if edge.target not in self._nodes:
                errors.append(f"Edge target {edge.target!r} not found")

        if errors:
            return errors

        # Detect cycles and distinguish valid guarded loops from infinite malformed cycles
        adj: dict[str, list[Edge]] = defaultdict(list)
        for e in self._edges:
            adj[e.source].append(e)

        # Find cycles using DFS path tracking
        visited: set[str] = set()
        path: list[str] = []
        path_set: set[str] = set()

        def _check_cycles(node: str) -> None:
            visited.add(node)
            path.append(node)
            path_set.add(node)

            for edge in adj.get(node, []):
                neighbor = edge.target
                if neighbor in path_set:
                    # Cycle found! Extract cycle path
                    idx = path.index(neighbor)
                    cycle_nodes = path[idx:] + [neighbor]

                    # Check if this cycle is guarded:
                    # A cycle is valid if ANY edge in the cycle has a condition,
                    # OR any node in the cycle has an outgoing edge exiting the cycle.
                    cycle_node_set = set(cycle_nodes)
                    has_exit_or_condition = False

                    for cn in cycle_node_set:
                        for outgoing_edge in adj.get(cn, []):
                            if outgoing_edge.is_conditional():
                                has_exit_or_condition = True
                                break
                            if outgoing_edge.target not in cycle_node_set:
                                has_exit_or_condition = True
                                break
                        if has_exit_or_condition:
                            break

                    if not has_exit_or_condition:
                        cycle_repr = " -> ".join(cycle_nodes)
                        errors.append(
                            f"Malformed infinite cycle detected: {cycle_repr} with no exit condition or guard."
                        )
                elif neighbor not in visited:
                    _check_cycles(neighbor)

            path.pop()
            path_set.remove(node)

        for n in self._nodes:
            if n not in visited:
                _check_cycles(n)

        return errors

    # -------------------------------------------------------------- Execution

    def run(
        self,
        initial_state: dict[str, Any] | GraphState | None = None,
        *,
        max_steps: int = 100,
        max_workers: int = 4,
    ) -> GraphResult:
        """Execute the graph from entry to completion."""
        errors = self.validate()
        if errors:
            if any("Malformed infinite cycle" in e for e in errors):
                raise GraphCycleError("; ".join(errors))
            raise GraphValidationError("; ".join(errors))

        state: GraphState = (
            initial_state.copy()
            if isinstance(initial_state, GraphState)
            else GraphState(initial_state or {})
        )

        traces: list[NodeTrace] = []
        node_visit_counts: dict[str, int] = defaultdict(int)

        # Adjacency and inbound edge mappings
        adj: dict[str, list[Edge]] = defaultdict(list)
        for e in self._edges:
            adj[e.source].append(e)

        # Barrier tracking: for join nodes, track pending inputs
        # Auto-detect join barriers: any node with multiple inbound edges is a barrier candidate
        inbound_count: dict[str, int] = defaultdict(int)
        for e in self._edges:
            inbound_count[e.target] += 1
        join_barriers = set(self._join_nodes) | {
            target for target, count in inbound_count.items() if count > 1
        }

        barrier_states: dict[str, list[GraphState]] = defaultdict(list)
        barrier_expected: dict[str, int] = defaultdict(int)

        t0 = time.perf_counter()
        steps = 0
        error: str | None = None

        # Execution queue of ready nodes
        # Each item is: node_name
        ready_queue: deque[str] = deque([self._entry])  # type: ignore[list-item]

        while ready_queue and steps < max_steps:
            # Batch of nodes ready to execute in parallel
            current_batch: list[str] = []
            while ready_queue:
                current_batch.append(ready_queue.popleft())

            # Check loop iterations on all nodes in this batch
            for node_name in current_batch:
                node_visit_counts[node_name] += 1
                if node_visit_counts[node_name] > self.max_iterations:
                    raise MaxIterationsExceededError(
                        f"Loop iteration limit ({self.max_iterations}) exceeded on node {node_name!r}"
                    )

            steps += len(current_batch)

            # Execute batch: in parallel if len > 1, else sequentially
            if len(current_batch) == 1:
                node_name = current_batch[0]
                node_trace, next_state = self._execute_node(self._nodes[node_name], state)
                traces.append(node_trace)
                if node_trace.error:
                    error = node_trace.error
                    break
                state = next_state
                # Resolve outgoing edges
                self._dispatch_outgoing(
                    node_name,
                    state,
                    adj,
                    join_barriers,
                    barrier_states,
                    barrier_expected,
                    ready_queue,
                )
            else:
                # Parallel fan-out
                with ThreadPoolExecutor(max_workers=min(len(current_batch), max_workers)) as pool:
                    futures = [
                        pool.submit(self._execute_node, self._nodes[name], state.copy())
                        for name in current_batch
                    ]
                    batch_results = [f.result() for f in futures]

                # Check errors & merge states
                for name, (node_trace, branch_state) in zip(current_batch, batch_results):
                    traces.append(node_trace)
                    if node_trace.error:
                        error = node_trace.error
                        break
                    # Merge branch state updates back into main state via reducers
                    state.merge(branch_state)

                    # Dispatch outgoing
                    self._dispatch_outgoing(
                        name,
                        branch_state,
                        adj,
                        join_barriers,
                        barrier_states,
                        barrier_expected,
                        ready_queue,
                    )

                if error is not None:
                    break

        total_ms = (time.perf_counter() - t0) * 1000
        success = error is None and all(t.error is None for t in traces)

        return GraphResult(
            state=state,
            trace=traces,
            nodes_executed=steps,
            total_ms=total_ms,
            graph_name=self.name,
            success=success,
            error=error,
        )

    def _execute_node(
        self,
        node: BaseNode,
        state: GraphState,
    ) -> tuple[NodeTrace, GraphState]:
        """Execute a single node safely with telemetry timing."""
        input_keys = list(state.keys())
        t_start = time.perf_counter()
        err: str | None = None
        new_state = state

        try:
            res = node.execute(state)
            if isinstance(res, GraphState):
                new_state = res
            elif isinstance(res, dict):
                new_state = state.copy()
                new_state.merge(res)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        t_end = time.perf_counter()
        output_keys = list(new_state.keys()) if err is None else input_keys

        trace = NodeTrace(
            node_name=node.name,
            node_type=node.node_type,
            started_at=t_start,
            ended_at=t_end,
            duration_ms=(t_end - t_start) * 1000,
            input_keys=input_keys,
            output_keys=output_keys,
            error=err,
        )
        return trace, new_state

    def _dispatch_outgoing(
        self,
        source: str,
        current_state: GraphState,
        adj: dict[str, list[Edge]],
        join_barriers: set[str],
        barrier_states: dict[str, list[GraphState]],
        barrier_expected: dict[str, int],
        ready_queue: deque[str],
    ) -> None:
        """Evaluate outgoing edges from source and schedule next nodes."""
        outgoing = adj.get(source, [])
        active_targets: list[str] = []

        for edge in outgoing:
            if edge.evaluate(current_state):
                active_targets.append(edge.target)

        for target in active_targets:
            if target in join_barriers:
                # Barrier synchronization:
                # Count inbound edges that lead to this barrier from current or sibling sources
                barrier_states[target].append(current_state)
                # Count total active predecessors leading to this barrier
                total_inbound = sum(1 for e in self._edges if e.target == target)
                if len(barrier_states[target]) >= total_inbound or len(active_targets) == 1:
                    # All incoming branches arrived at barrier
                    if target not in ready_queue:
                        ready_queue.append(target)
            else:
                if target not in ready_queue:
                    ready_queue.append(target)

    # ---------------------------------------------------------- Serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph definition to JSON dict (ReactFlow compatible)."""
        return {
            "name": self.name,
            "description": self.description,
            "entry": self._entry,
            "max_iterations": self.max_iterations,
            "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
            "edges": [edge.to_dict() for edge in self._edges],
        }


# Fluent / LangGraph style aliases
FlowGraph = AgentGraph
FlowState = GraphState
