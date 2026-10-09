"""Unit tests for dynaflow agent graph engine and control flow (Issues #277 & #278)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from dynavec.agents import (
    AgentGraph,
    BaseNode,
    GraphCycleError,
    GraphResult,
    GraphState,
    GraphValidationError,
    MaxIterationsExceededError,
    append_reducer,
)

# ---------------------------------------------------------------------------
# Test Helpers & Custom Nodes
# ---------------------------------------------------------------------------


class IncrementNode(BaseNode):
    """Simple node that increments a counter in state."""

    node_type = "increment"

    def __init__(self, key: str = "count", amount: int = 1, name: str = "") -> None:
        super().__init__(name=name)
        self.key = key
        self.amount = amount

    def execute(self, state: GraphState) -> dict[str, Any]:
        return {self.key: state.get(self.key, 0) + self.amount}


# ---------------------------------------------------------------------------
# 1. Core End-to-End Execution (Issue #277)
# ---------------------------------------------------------------------------


def test_linear_three_node_graph_execution() -> None:
    """Acceptance criteria #277: A graph of >=3 nodes executes end-to-end with state passed."""

    def step1(state: GraphState) -> dict[str, Any]:
        return {"query": state["query"].strip().lower()}

    def step2(state: GraphState) -> dict[str, Any]:
        return {"tokens": state["query"].split()}

    def step3(state: GraphState) -> dict[str, Any]:
        return {"token_count": len(state["tokens"])}

    graph = (
        AgentGraph("test-linear")
        .add_node("clean", step1)
        .add_node("tokenize", step2)
        .add_node("count", step3)
        .add_edge("clean", "tokenize")
        .add_edge("tokenize", "count")
        .set_entry("clean")
    )

    result = graph.run({"query": "  Dynavec Vector DB  "})

    assert isinstance(result, GraphResult)
    assert result.success is True
    assert result.nodes_executed == 3
    assert result.state["query"] == "dynavec vector db"
    assert result.state["tokens"] == ["dynavec", "vector", "db"]
    assert result.state["token_count"] == 3
    assert len(result.trace) == 3


def test_state_reducers_and_isolation() -> None:
    """Acceptance criteria #277: State updates are isolated and merged via defined reducers."""
    state = GraphState(
        {"items": ["a"], "count": 10},
        reducers={
            "items": append_reducer,
            "count": lambda a, b: a + b,
        },
    )

    def node_append(s: GraphState) -> dict[str, Any]:
        return {"items": ["b", "c"], "count": 5}

    graph = AgentGraph("test-reducers").add_node("append", node_append).set_entry("append")

    result = graph.run(state)
    assert result.success is True
    # items merged via append_reducer
    assert result.state["items"] == ["a", "b", "c"]
    # count merged via sum reducer
    assert result.state["count"] == 15


# ---------------------------------------------------------------------------
# 2. Conditional Routing & Branches (Issue #278)
# ---------------------------------------------------------------------------


def test_conditional_edge_branch_on_predicate() -> None:
    """Acceptance criteria #278: A graph can branch on a predicate."""
    graph = (
        AgentGraph("test-branching")
        .add_node("start", lambda s: {"val": s.get("input", 0)})
        .add_node("high_path", lambda s: {"route": "HIGH", "score": s["val"] * 2})
        .add_node("low_path", lambda s: {"route": "LOW", "score": s["val"] // 2})
        .add_edge("start", "high_path", condition="val >= 10")
        .add_edge("start", "low_path", condition="val < 10")
        .set_entry("start")
    )

    # Test high path
    res_high = graph.run({"input": 15})
    assert res_high.success is True
    assert res_high.state["route"] == "HIGH"
    assert res_high.state["score"] == 30
    assert [t.node_name for t in res_high.trace] == ["start", "high_path"]

    # Test low path
    res_low = graph.run({"input": 4})
    assert res_low.success is True
    assert res_low.state["route"] == "LOW"
    assert res_low.state["score"] == 2
    assert [t.node_name for t in res_low.trace] == ["start", "low_path"]


def test_conditional_edge_callable_predicate() -> None:
    """Verify conditional edges evaluate Python callable predicates."""
    graph = (
        AgentGraph("test-callable-edge")
        .add_node("eval", lambda s: {"score": 0.85})
        .add_node("pass_node", lambda s: {"verdict": "APPROVED"})
        .add_node("fail_node", lambda s: {"verdict": "REJECTED"})
        .add_edge("eval", "pass_node", condition=lambda s: s["score"] >= 0.8)
        .add_edge("eval", "fail_node", condition=lambda s: s["score"] < 0.8)
        .set_entry("eval")
    )

    result = graph.run()
    assert result.state["verdict"] == "APPROVED"
    assert [t.node_name for t in result.trace] == ["eval", "pass_node"]


# ---------------------------------------------------------------------------
# 3. Parallel Branches (Fan-Out + Join Barrier) (Issue #278)
# ---------------------------------------------------------------------------


def test_parallel_branches_fan_out_and_join() -> None:
    """Acceptance criteria #278: Run two branches in parallel and join."""

    def branch_a(s: GraphState) -> dict[str, Any]:
        time.sleep(0.01)
        return {"results": ["from_A"]}

    def branch_b(s: GraphState) -> dict[str, Any]:
        time.sleep(0.01)
        return {"results": ["from_B"]}

    def join_node(s: GraphState) -> dict[str, Any]:
        return {"joined": True, "total_results": len(s.get("results", []))}

    state = GraphState({"results": []}, reducers={"results": append_reducer})

    graph = (
        AgentGraph("test-parallel-join")
        .add_node("start", lambda s: {"initialized": True})
        .add_node("branch_a", branch_a)
        .add_node("branch_b", branch_b)
        .add_node("join", join_node)
        .add_parallel_branches("start", ["branch_a", "branch_b"], join_at="join")
        .set_entry("start")
    )

    result = graph.run(state)

    assert result.success is True
    assert result.state["joined"] is True
    assert sorted(result.state["results"]) == ["from_A", "from_B"]
    assert result.state["total_results"] == 2
    executed_nodes = [t.node_name for t in result.trace]
    assert "start" == executed_nodes[0]
    assert "join" == executed_nodes[-1]
    assert set(executed_nodes[1:3]) == {"branch_a", "branch_b"}


# ---------------------------------------------------------------------------
# 4. Loops & Cycles with Termination Guards (Issue #278)
# ---------------------------------------------------------------------------


def test_bounded_loop_terminates_on_condition() -> None:
    """Acceptance criteria #278: A loop terminates on condition and respects iteration cap."""
    # Graph that increments counter in a loop until count >= 3
    graph = (
        AgentGraph("test-loop-termination", max_iterations=10)
        .add_node("init", lambda s: {"count": 0})
        .add_node("step", IncrementNode("count", amount=1, name="step"))
        .add_node("done", lambda s: {"status": "FINISHED"})
        .add_edge("init", "step")
        # Loop back if count < 3
        .add_edge("step", "step", condition="count < 3")
        # Exit to done when count >= 3
        .add_edge("step", "done", condition="count >= 3")
        .set_entry("init")
    )

    result = graph.run()

    assert result.success is True
    assert result.state["count"] == 3
    assert result.state["status"] == "FINISHED"
    assert [t.node_name for t in result.trace] == [
        "init",
        "step",
        "step",
        "step",
        "done",
    ]


def test_loop_exceeds_max_iterations_raises_error() -> None:
    """Acceptance criteria #278: Respects iteration cap and halts runaway loops."""
    # Never terminates because target is impossible
    graph = (
        AgentGraph("test-infinite-loop", max_iterations=5)
        .add_node("step", IncrementNode("count", 1, name="step"))
        .add_node("exit", lambda s: {"done": True})
        .add_edge("step", "step", condition="count < 100")
        .add_edge("step", "exit", condition="count >= 100")
        .set_entry("step")
    )

    with pytest.raises(MaxIterationsExceededError) as exc_info:
        graph.run({"count": 0})

    assert "exceeded" in str(exc_info.value).lower()
    assert "step" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Cycle Detection & Validation Errors (Issue #278)
# ---------------------------------------------------------------------------


def test_cycle_detection_flags_unconditional_infinite_cycle() -> None:
    """Acceptance criteria #278: Cycle detection and clear errors for malformed graphs."""
    # Node A -> Node B -> Node A with no condition or exit
    graph = (
        AgentGraph("test-malformed-cycle")
        .add_node("node_a", lambda s: s)
        .add_node("node_b", lambda s: s)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", "node_a")  # Unconditional back-edge
        .set_entry("node_a")
    )

    with pytest.raises(GraphCycleError) as exc_info:
        graph.run()

    assert "Malformed infinite cycle detected" in str(exc_info.value)
    assert "node_a" in str(exc_info.value)
    assert "node_b" in str(exc_info.value)


def test_graph_validation_catches_missing_entry_or_targets() -> None:
    """Verify validation catches missing nodes or disconnected entries."""
    graph = AgentGraph("invalid-graph")
    graph.add_node("n1", lambda s: s)

    # Missing entry
    with pytest.raises(GraphValidationError) as exc:
        graph.run()
    assert "No entry node set" in str(exc.value)

    # Non-existent target in edge
    graph.set_entry("n1")
    graph.add_edge("n1", "missing_node")
    with pytest.raises(GraphValidationError) as exc2:
        graph.run()
    assert "Edge target 'missing_node' not found" in str(exc2.value)


# ---------------------------------------------------------------------------
# 6. Serialization (ReactFlow Compatibility)
# ---------------------------------------------------------------------------


def test_graph_serialization_to_dict() -> None:
    """Verify graph serialization to dictionary format compatible with UI canvas."""
    graph = (
        AgentGraph("serialize-test", description="Demo workflow")
        .add_node("fetch", lambda s: s)
        .add_node("format", lambda s: s)
        .add_edge("fetch", "format", condition="ready == True", label="on-ready")
        .set_entry("fetch")
    )

    serialized = graph.to_dict()
    assert serialized["name"] == "serialize-test"
    assert serialized["entry"] == "fetch"
    assert "fetch" in serialized["nodes"]
    assert "format" in serialized["nodes"]
    assert len(serialized["edges"]) == 1
    assert serialized["edges"][0]["source"] == "fetch"
    assert serialized["edges"][0]["target"] == "format"
    assert serialized["edges"][0]["condition"] == "ready == True"
    assert serialized["edges"][0]["label"] == "on-ready"
