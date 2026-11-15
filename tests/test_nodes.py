"""Unit tests for dynaflow built-in node library and graph execution pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from dynavec import (
    GRAPH_SCHEMA_VERSION,
    ApprovalNode,
    CodeNode,
    Edge,
    GraphRunState,
    LLMNode,
    LoopNode,
    Node,
    NodePort,
    NodeRegistry,
    NodeSchema,
    PortCompatibilityError,
    RetrieverNode,
    RetryPolicy,
    RouterNode,
    SimpleGraph,
    SubGraphNode,
    ToolNode,
    WebhookNode,
    custom_node,
    is_port_type_compatible,
    node_registry,
)
from dynavec.chat.base import ChatChunk, ChatModel, ChatResult, Message


class MockChatModel(ChatModel):
    """Simple mock chat model for testing LLMNode."""

    def invoke(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].content
        return ChatResult(
            message=Message(
                role="assistant",
                content=f"Processed: {last}",
            ),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        last = messages[-1].content
        yield ChatChunk(content=f"Processed: {last}")


class MockDoc:
    def __init__(self, doc_id: str, text: str, score: float = 0.9) -> None:
        self.id = doc_id
        self.text = text
        self.score = score
        self.metadata = {"source": "unit-test"}


# =====================================================================
# Registry & Schema Tests
# =====================================================================


def test_registry_registration_and_palette() -> None:
    expected_types = {
        "approval",
        "code",
        "llm",
        "loop",
        "retriever",
        "router",
        "subgraph",
        "tool",
        "webhook",
    }
    registered = set(node_registry.list_types())
    assert expected_types.issubset(registered)

    palette = node_registry.export_canvas_palette()
    palette_types = {item["type"] for item in palette}
    assert expected_types.issubset(palette_types)

    # Check palette item structure
    for item in palette:
        assert "type" in item
        assert "label" in item
        assert "category" in item
        assert "inputPorts" in item
        assert "outputPorts" in item


def test_port_and_schema_serialization() -> None:
    port = NodePort(
        name="test_in",
        port_type="string",
        description="A test port",
        required=True,
        default="val",
    )
    p_dict = port.to_dict()
    assert p_dict["name"] == "test_in"
    assert p_dict["port_type"] == "string"
    assert p_dict["required"] is True
    assert p_dict["default"] == "val"

    schema = NodeSchema(
        node_type="custom_test",
        label="Custom Test",
        description="A test node schema",
        category="testing",
        inputs=[port],
        outputs=[NodePort(name="test_out", port_type="number")],
        config_schema={"timeout": {"type": "integer", "default": 30}},
    )

    s_dict = schema.to_dict()
    assert s_dict["node_type"] == "custom_test"
    assert len(s_dict["inputs"]) == 1
    assert len(s_dict["outputs"]) == 1

    rf_dict = schema.to_reactflow_dict()
    assert rf_dict["type"] == "custom_test"
    assert rf_dict["inputPorts"] == ["test_in"]
    assert rf_dict["outputPorts"] == ["test_out"]
    assert rf_dict["config"]["timeout"]["default"] == 30


# =====================================================================
# Individual Built-in Node Tests
# =====================================================================


def test_retriever_node() -> None:
    def dummy_retriever(
        query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[MockDoc]:
        return [
            MockDoc("d1", "Dynavec is a serverless vector database."),
            MockDoc("d2", "DynamoDB provides scalable key-value storage."),
        ]

    node = RetrieverNode(retriever_fn=dummy_retriever, config={"top_k": 2})
    assert node.node_type == "retriever"

    rf_node = node.to_dict()
    assert rf_node["type"] == "retriever"
    assert rf_node["data"]["category"] == "retrieval"

    outputs = node.execute({"query": "What is Dynavec?"})
    assert len(outputs["documents"]) == 2
    assert outputs["documents"][0]["id"] == "d1"
    assert "Dynavec is a serverless" in outputs["context"]
    assert "DynamoDB provides" in outputs["context"]


def test_llm_node_chat_model() -> None:
    model = MockChatModel()
    node = LLMNode(model=model, config={"system_prompt": "You are a helpful assistant."})

    outputs = node.execute(
        {
            "prompt": "Explain vector indexing",
            "context": "Context snippet here.",
        }
    )
    assert "response" in outputs
    assert (
        "Processed: Context:\nContext snippet here.\n\nUser Question:\nExplain vector indexing"
        in outputs["response"]
    )
    assert outputs["message"].role == "assistant"


def test_llm_node_callable() -> None:
    def simple_callable(messages: list[Message]) -> str:
        return f"Echo: {messages[-1].content}"

    node = LLMNode(model=simple_callable)
    outputs = node.execute({"prompt": "Hello world"})
    assert outputs["response"] == "Echo: Hello world"


def test_tool_node_success() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    node = ToolNode(tool=add)
    outputs = node.execute({"arguments": {"a": 10, "b": 25}})
    assert outputs["status"] == "success"
    assert outputs["result"] == 35


def test_tool_node_json_string_and_error() -> None:
    def multiply(x: int, y: int) -> int:
        return x * y

    node = ToolNode(tool=multiply)
    outputs = node.execute({"arguments": '{"x": 6, "y": 7}'})
    assert outputs["status"] == "success"
    assert outputs["result"] == 42

    # Test error handling
    fail_node = ToolNode(tool=lambda: 1 / 0)
    err_out = fail_node.execute({})
    assert err_out["status"] == "error"
    assert "division by zero" in err_out["result"]


def test_router_node() -> None:
    node = RouterNode(route_fn=lambda x: "admin_flow" if x.get("is_admin") else "user_flow")

    out1 = node.execute({"data": {"is_admin": True}})
    assert out1["selected_route"] == "admin_flow"
    assert out1["data"] == {"is_admin": True}

    out2 = node.execute({"data": {"is_admin": False}})
    assert out2["selected_route"] == "user_flow"

    # Explicit route override
    out3 = node.execute({"data": {}, "route": "manual_override"})
    assert out3["selected_route"] == "manual_override"


def test_code_node() -> None:
    def custom_transform(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"uppercased": inputs["text"].upper(), "len": len(inputs["text"])}

    node = CodeNode(code_fn=custom_transform)
    res = node.execute({"inputs": {"text": "dynaflow"}})
    assert res["uppercased"] == "DYNAFLOW"
    assert res["len"] == 8

    # When code returns non-dict
    scalar_node = CodeNode(code_fn=lambda inp: {"outputs": len(inp["text"])})
    res2 = scalar_node.execute({"inputs": {"text": "test"}})
    assert res2["outputs"] == 4


def test_approval_node() -> None:
    node = ApprovalNode()
    out = node.execute(
        {
            "request": {"action": "delete_database", "target": "prod"},
            "approved": True,
            "notes": "Approved by SecOps",
        }
    )
    assert out["approved"] is True
    assert out["data"]["action"] == "delete_database"
    assert out["notes"] == "Approved by SecOps"

    out_rejected = node.execute(
        {
            "request": {"action": "drop_table"},
            "approved": False,
        }
    )
    assert out_rejected["approved"] is False


def test_loop_node() -> None:
    node = LoopNode(map_fn=lambda x: x * 2, config={"max_iterations": 3})
    out = node.execute({"items": [1, 2, 3, 4, 5]})
    assert out["items"] == [2, 4, 6]  # clipped to max 3
    assert out["count"] == 3


def test_webhook_node() -> None:
    node = WebhookNode()
    payload = {"event": "user.signup", "user_id": "usr_123"}
    out = node.execute({"payload": payload, "headers": {"Authorization": "Bearer tok"}})
    assert out["payload"] == payload


def test_subgraph_node() -> None:
    def nested_workflow(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"processed_by_subgraph": True, "value": inputs.get("val", 0) + 10}

    node = SubGraphNode(subgraph=nested_workflow)
    out = node.execute({"inputs": {"val": 5}})
    assert out["outputs"]["processed_by_subgraph"] is True
    assert out["outputs"]["value"] == 15


# =====================================================================
# SimpleGraph Pipeline Integration: Retriever -> LLM -> Tool
# =====================================================================


def test_simple_graph_pipeline_retriever_llm_tool() -> None:
    # 1. RetrieverNode: searches knowledge base
    def mock_retriever(
        query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[MockDoc]:
        return [MockDoc("doc-1", "Dynavec provides DynamoDB vectors.")]

    retriever = RetrieverNode(retriever_fn=mock_retriever, node_id="retriever_1")

    # 2. LLMNode: formats tool call request based on context
    def mock_llm(messages: list[Message]) -> str:
        # LLM decides to call calculator tool with argument 42
        return '{"operation": "double", "value": 42}'

    llm = LLMNode(model=mock_llm, node_id="llm_1")

    # 3. ToolNode: executes calculation
    def mock_tool(operation: str, value: int) -> int:
        if operation == "double":
            return value * 2
        return value

    tool = ToolNode(tool=mock_tool, node_id="tool_1")

    # Assemble graph
    graph = SimpleGraph()
    graph.add_node(retriever)
    graph.add_node(llm)
    graph.add_node(tool)

    # retriever.context -> llm.context
    graph.add_edge(
        source_id="retriever_1",
        target_id="llm_1",
        source_port="context",
        target_port="context",
    )

    # llm.response -> tool.arguments
    graph.add_edge(
        source_id="llm_1",
        target_id="tool_1",
        source_port="response",
        target_port="arguments",
    )

    result = graph.run({"query": "How many vectors?", "prompt": "Calculate vector capacity"})

    assert "documents" in result
    assert "DynamoDB vectors" in result["context"]
    assert result["status"] == "success"
    assert result["result"] == 84


def test_simple_graph_validation_errors() -> None:
    graph = SimpleGraph()
    with pytest.raises(ValueError, match="Source node 'nonexistent' not in graph"):
        graph.add_edge("nonexistent", "target")

    node = CodeNode(code_fn=lambda x: x, node_id="valid_node")
    graph.add_node(node)
    with pytest.raises(ValueError, match="Target node 'missing_target' not in graph"):
        graph.add_edge("valid_node", "missing_target")


# =====================================================================
# Engine Resilience, Retries & Resumability Tests (Issue #279)
# =====================================================================


def test_retry_policy_logic_and_serialization() -> None:
    policy = RetryPolicy(
        max_attempts=4,
        base_delay=0.01,
        max_delay=0.1,
        jitter=False,
        retry_on=(ValueError, KeyError),
    )
    assert policy.should_retry(ValueError("bad value"))
    assert policy.should_retry(KeyError("missing key"))
    assert not policy.should_retry(TypeError("wrong type"))

    # Serialization
    d = policy.to_dict()
    assert d["max_attempts"] == 4
    assert d["base_delay"] == 0.01
    assert not d["jitter"]

    reconstructed = RetryPolicy.from_dict(d)
    assert reconstructed.max_attempts == 4
    assert reconstructed.base_delay == 0.01


def test_node_retry_transient_failure_success() -> None:
    call_count = 0

    def flaky_fn(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary network glitch")
        return {"result": inputs["x"] * 10}

    policy = RetryPolicy(max_attempts=4, base_delay=0.001, max_delay=0.01)
    flaky_node = CodeNode(
        code_fn=flaky_fn, node_id="flaky_node", retry_policy=policy
    )

    graph = SimpleGraph()
    graph.add_node(flaky_node)

    res = graph.run({"x": 5})
    assert res["result"] == 50
    assert flaky_node.attempts == 3
    assert flaky_node.status == "succeeded"
    assert flaky_node.last_error is None


def test_node_retry_exhaustion_failure() -> None:
    attempts_recorded = 0

    def always_failing_fn(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts_recorded
        attempts_recorded += 1
        raise RuntimeError("Hard database failure")

    policy = RetryPolicy(max_attempts=3, base_delay=0.001, max_delay=0.01)
    failing_node = CodeNode(
        code_fn=always_failing_fn, node_id="fail_node", retry_policy=policy
    )

    graph = SimpleGraph()
    graph.add_node(failing_node)

    with pytest.raises(RuntimeError, match="Hard database failure"):
        graph.run({"x": 1}, raise_on_failure=True)

    assert failing_node.attempts == 3
    assert failing_node.status == "failed"
    assert "Hard database failure" in (failing_node.last_error or "")


def test_error_edge_fallback_routing() -> None:
    def failing_primary_fn(inputs: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Primary service unavailable")

    def fallback_recovery_fn(inputs: dict[str, Any]) -> dict[str, Any]:
        error_msg = inputs.get("error", "")
        return {
            "recovered": True,
            "handled_error": error_msg,
            "fallback_response": "Served from cached backup",
        }

    primary = CodeNode(
        code_fn=failing_primary_fn,
        node_id="primary_node",
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.001),
    )
    fallback = CodeNode(code_fn=fallback_recovery_fn, node_id="fallback_node")

    graph = SimpleGraph()
    graph.add_node(primary)
    graph.add_node(fallback)

    # Wire error edge
    graph.add_error_edge(source_id="primary_node", target_id="fallback_node")

    res = graph.run({"query": "fetch stats"}, raise_on_failure=False)
    assert res["recovered"] is True
    assert "Primary service unavailable" in res["handled_error"]
    assert res["fallback_response"] == "Served from cached backup"
    assert primary.status == "failed"
    assert fallback.status == "succeeded"


def test_graph_run_state_checkpoint_serialization() -> None:
    state = GraphRunState(
        run_id="run_test_123",
        initial_inputs={"a": 1},
        current_state={"a": 1, "b": 2},
        node_outputs={"node_1": {"b": 2}},
        node_statuses={"node_1": "succeeded", "node_2": "pending"},
        node_errors={"node_3": "some error"},
        completed=False,
        failed_node="node_3",
    )

    serialized = state.to_dict()
    assert serialized["run_id"] == "run_test_123"
    assert serialized["completed"] is False
    assert serialized["node_statuses"]["node_1"] == "succeeded"

    restored = GraphRunState.from_dict(serialized)
    assert restored.run_id == "run_test_123"
    assert restored.current_state == {"a": 1, "b": 2}
    assert restored.node_outputs == {"node_1": {"b": 2}}
    assert restored.node_statuses == {"node_1": "succeeded", "node_2": "pending"}


def test_graph_resumability_from_checkpoint() -> None:
    # 3-node linear pipeline: Node A -> Node B (initially fails) -> Node C
    node_a_runs = 0
    node_b_runs = 0
    node_c_runs = 0

    def fn_a(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal node_a_runs
        node_a_runs += 1
        return {"step_a": "done", "value": inputs["start"] + 10}

    def fn_b_fail(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal node_b_runs
        node_b_runs += 1
        raise ValueError("Simulated outage in Node B")

    def fn_b_fixed(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal node_b_runs
        node_b_runs += 1
        return {"step_b": "done", "value": inputs["value"] * 2}

    def fn_c(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal node_c_runs
        node_c_runs += 1
        return {"step_c": "done", "final_result": inputs["value"] + 5}

    node_a = CodeNode(code_fn=fn_a, node_id="node_a")
    node_b = CodeNode(code_fn=fn_b_fail, node_id="node_b")
    node_c = CodeNode(code_fn=fn_c, node_id="node_c")

    graph = SimpleGraph()
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")

    # Run 1: Fails at Node B
    state_result = graph.run({"start": 5}, return_state=True, raise_on_failure=False)
    assert isinstance(state_result, GraphRunState)
    assert state_result.completed is False
    assert state_result.failed_node == "node_b"
    assert node_a_runs == 1
    assert node_b_runs == 1
    assert node_c_runs == 0

    # Save checkpoint
    checkpoint = state_result.to_dict()

    # Fix Node B
    node_b.execute = fn_b_fixed  # type: ignore[method-assign]

    # Resume from checkpoint
    resumed_state = graph.resume(checkpoint)

    # Verify Node A was NOT re-executed
    assert node_a_runs == 1  # Crucial: stayed 1, skipped duplicate run!
    assert node_b_runs == 2  # executed upon resume
    assert node_c_runs == 1  # executed to completion

    assert resumed_state.completed is True
    assert resumed_state.failed_node is None
    assert resumed_state.current_state["step_a"] == "done"
    assert resumed_state.current_state["step_b"] == "done"
    assert resumed_state.current_state["step_c"] == "done"
    # start (5) + 10 = 15; 15 * 2 = 30; 30 + 5 = 35
    assert resumed_state.current_state["final_result"] == 35


# =====================================================================
# Issue #282: Typed Node System, Schema, Port Compatibility & JSON Round-Trip
# =====================================================================


def test_port_type_compatibility_rules() -> None:
    # "any" is compatible with any type
    assert is_port_type_compatible("any", "string") is True
    assert is_port_type_compatible("number", "any") is True
    assert is_port_type_compatible("any", "any") is True

    # Exact matches
    assert is_port_type_compatible("string", "string") is True
    assert is_port_type_compatible("number", "number") is True
    assert is_port_type_compatible("boolean", "boolean") is True
    assert is_port_type_compatible("array", "array") is True
    assert is_port_type_compatible("object", "object") is True

    # JSON serialization / deserialization bridge
    assert is_port_type_compatible("string", "object") is True
    assert is_port_type_compatible("object", "string") is True

    # Incompatible pairings
    assert is_port_type_compatible("number", "string") is False
    assert is_port_type_compatible("boolean", "number") is False
    assert is_port_type_compatible("array", "boolean") is False
    assert is_port_type_compatible("number", "object") is False


def test_edge_serialization_round_trip() -> None:
    edge = Edge(
        source_node="node_1",
        target_node="node_2",
        source_port="out_data",
        target_port="in_data",
        mapping={"key1": "key2"},
        is_error_edge=False,
    )
    data = edge.to_dict()
    assert data["source"] == "node_1"
    assert data["target"] == "node_2"
    assert data["source_port"] == "out_data"
    assert data["target_port"] == "in_data"
    assert data["mapping"] == {"key1": "key2"}
    assert data["is_error_edge"] is False

    restored = Edge.from_dict(data)
    assert restored.source_node == "node_1"
    assert restored.target_node == "node_2"
    assert restored.source_port == "out_data"
    assert restored.target_port == "in_data"
    assert restored.mapping == {"key1": "key2"}
    assert restored.is_error_edge is False


def test_port_compatibility_validation_in_graph() -> None:
    # Node with declared schema
    schema_producer = NodeSchema(
        node_type="num_producer",
        label="Number Producer",
        description="Emits numbers",
        outputs=[NodePort(name="val", port_type="number")],
    )
    schema_str_consumer = NodeSchema(
        node_type="str_consumer",
        label="String Consumer",
        description="Requires strings",
        inputs=[NodePort(name="text", port_type="string")],
    )

    class NumNode(Node):
        node_type = "num_producer"
        schema = schema_producer

        def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
            return {"val": 42}

    class StrNode(Node):
        node_type = "str_consumer"
        schema = schema_str_consumer

        def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
            return {"received": inputs["text"]}

    graph = SimpleGraph()
    graph.add_node(NumNode(node_id="n1"))
    graph.add_node(StrNode(node_id="n2"))

    # Direct incompatible port connection should raise PortCompatibilityError
    with pytest.raises(PortCompatibilityError, match="Cannot connect incompatible ports"):
        graph.add_edge("n1", "n2", source_port="val", target_port="text")

    # Mapping with incompatible ports should also raise
    with pytest.raises(PortCompatibilityError, match="Cannot connect incompatible ports"):
        graph.add_edge("n1", "n2", mapping={"val": "text"})

    # Connecting undeclared port should raise ValueError
    with pytest.raises(ValueError, match="Source port 'unknown_out' not declared"):
        graph.add_edge("n1", "n2", source_port="unknown_out", target_port="text")

    with pytest.raises(ValueError, match="Target port 'unknown_in' not declared"):
        graph.add_edge("n1", "n2", source_port="val", target_port="unknown_in")

    # Disabling validate_ports allows connection without raising
    graph.add_edge("n1", "n2", source_port="val", target_port="text", validate_ports=False)

    # Calling graph.validate() checks all edges and catches the incompatibility
    with pytest.raises(PortCompatibilityError, match="Cannot connect incompatible ports"):
        graph.validate()


def test_graph_json_serialization_round_trip() -> None:
    # Build a multi-node graph with CodeNode and RouterNode
    def step1(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"query": inputs.get("query", "").strip(), "count": 10}

    def step2(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"formatted": f"Processed: {inputs.get('query')} (x{inputs.get('count')})"}

    node1 = CodeNode(code_fn=step1, node_id="step1", config={"opt": True})
    node2 = CodeNode(code_fn=step2, node_id="step2")

    graph = SimpleGraph()
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("step1", "step2")

    # Serialize to dict and JSON string
    graph_dict = graph.to_dict()
    assert graph_dict["version"] == GRAPH_SCHEMA_VERSION
    assert len(graph_dict["nodes"]) == 2
    assert len(graph_dict["edges"]) == 1
    assert graph_dict["execution_order"] == ["step1", "step2"]

    json_str = graph.to_json()
    assert f'"version": "{GRAPH_SCHEMA_VERSION}"' in json_str

    # Deserialize back from JSON
    restored_graph = SimpleGraph.from_json(json_str)
    assert len(restored_graph.nodes) == 2
    assert len(restored_graph.edges) == 1
    assert restored_graph._execution_order == ["step1", "step2"]

    # Verify node config and type preserved
    assert restored_graph.nodes["step1"].config == {"opt": True}
    assert restored_graph.nodes["step1"].node_type == "code"
    assert restored_graph.nodes["step2"].node_type == "code"

    # Edge preserved
    edge = restored_graph.edges[0]
    assert edge.source_node == "step1"
    assert edge.target_node == "step2"

    # Verify schema version validation
    with pytest.raises(ValueError, match="Unsupported graph schema version '99.0'"):
        SimpleGraph.from_dict({"version": "99.0", "nodes": [], "edges": []})

    with pytest.raises(ValueError, match="Missing 'version'"):
        SimpleGraph.from_dict({"nodes": [], "edges": []})

    # Unknown node type validation
    with pytest.raises(ValueError, match="Unknown node type 'unregistered_type'"):
        SimpleGraph.from_dict({
            "version": GRAPH_SCHEMA_VERSION,
            "nodes": [{"id": "x", "type": "unregistered_type", "data": {}}],
            "edges": [],
        })


# =====================================================================
# Issue #284: Custom Nodes via SDK & Registry
# =====================================================================


def test_custom_node_subclass_and_registration() -> None:
    # 1. Define custom node via class inheritance and @node_registry.register
    @node_registry.register
    class MultiplierCustomNode(Node):
        node_type = "multiplier_custom_test"
        schema = NodeSchema(
            node_type="multiplier_custom_test",
            label="Multiplier Node",
            description="Multiplies inputs by a configured factor",
            category="math",
            inputs=[NodePort(name="val", port_type="number")],
            outputs=[NodePort(name="result", port_type="number")],
            config_schema={"factor": {"type": "number", "default": 2}},
        )

        def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
            factor = self.config.get("factor", 2)
            return {"result": inputs["val"] * factor}

    try:
        # Check registration
        assert node_registry.has("multiplier_custom_test") is True
        assert "multiplier_custom_test" in node_registry.list_types()
        assert node_registry.get("multiplier_custom_test") is MultiplierCustomNode

        # Check canvas palette includes custom node
        palette = node_registry.export_canvas_palette()
        custom_item = next(
            (p for p in palette if p["type"] == "multiplier_custom_test"), None
        )
        assert custom_item is not None
        assert custom_item["label"] == "Multiplier Node"
        assert custom_item["category"] == "math"
        assert custom_item["inputPorts"] == ["val"]
        assert custom_item["outputPorts"] == ["result"]

        # Instantiate and run standalone
        node = MultiplierCustomNode(config={"factor": 5})
        output = node.execute({"val": 10})
        assert output["result"] == 50

        # Wire into a SimpleGraph and run
        graph = SimpleGraph()
        graph.add_node(node)
        res = graph.run({"val": 7})
        assert res["result"] == 35

        # Verify graph round-trip with custom node
        json_repr = graph.to_json()
        reloaded_graph = SimpleGraph.from_json(json_repr)
        reloaded_node = reloaded_graph.nodes[node.id]
        assert isinstance(reloaded_node, MultiplierCustomNode)
        assert reloaded_node.config == {"factor": 5}
        assert reloaded_graph.run({"val": 6})["result"] == 30

    finally:
        node_registry.unregister("multiplier_custom_test")
        assert node_registry.has("multiplier_custom_test") is False


def test_custom_node_decorator_and_round_trip() -> None:
    # 2. Define custom node using the @custom_node decorator
    custom_reg = NodeRegistry()

    @custom_node(
        node_type="text_sentiment",
        label="Text Sentiment Analyzer",
        description="Analyzes polarity score of input text",
        category="nlp",
        inputs=[("text", "string")],
        outputs=[("sentiment", "string"), ("confidence", "number")],
        config_schema={"threshold": {"type": "number", "default": 0.5}},
        registry=custom_reg,
    )
    def analyze_sentiment(
        inputs: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        text = inputs.get("text", "")
        threshold = config.get("threshold", 0.5)
        pos = "good" in text.lower() or "excellent" in text.lower()
        return {
            "sentiment": "positive" if pos else "neutral",
            "confidence": 0.95 if pos else threshold,
        }

    # Verify custom registry holds the node
    assert custom_reg.has("text_sentiment") is True
    schema = custom_reg.get_schema("text_sentiment")
    assert schema is not None
    assert schema.node_type == "text_sentiment"
    assert schema.category == "nlp"
    assert len(schema.inputs) == 1
    assert schema.inputs[0].name == "text"
    assert schema.inputs[0].port_type == "string"
    assert len(schema.outputs) == 2
    assert schema.outputs[0].name == "sentiment"
    assert schema.outputs[1].name == "confidence"

    # Instantiate custom node
    sentiment_node_instance = analyze_sentiment(
        node_id="sentiment_1", config={"threshold": 0.6}
    )
    out = sentiment_node_instance.execute({"text": "This vector database is excellent!"})
    assert out["sentiment"] == "positive"
    assert out["confidence"] == 0.95

    # Test graph composition with custom registry
    graph = SimpleGraph()
    graph.add_node(sentiment_node_instance)
    res = graph.run({"text": "Average response"})
    assert res["sentiment"] == "neutral"
    assert res["confidence"] == 0.6

    # Graph JSON round-trip using custom registry
    json_data = graph.to_json()
    graph_reconstructed = SimpleGraph.from_json(json_data, registry=custom_reg)
    res2 = graph_reconstructed.run({"text": "This is good"})
    assert res2["sentiment"] == "positive"
    assert res2["confidence"] == 0.95


def test_custom_node_pipeline_with_builtin_nodes() -> None:
    # 3. End-to-end integration: RetrieverNode -> LLMNode -> CustomFormatterNode
    custom_type = "summary_formatter_test"

    @custom_node(
        node_type=custom_type,
        label="Summary Formatter",
        description="Formats response text into a markdown report",
        inputs=[("response", "string")],
        outputs=[("report", "string")],
        config_schema={"header": {"type": "string", "default": "Dynavec Report"}},
    )
    def format_summary(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        header = config.get("header", "Dynavec Report")
        body = inputs.get("response", "")
        return {"report": f"# {header}\n\n{body}"}

    try:
        def dummy_retriever(query: str, **kwargs: Any) -> list[MockDoc]:
            return [MockDoc("d100", "S3 Vectors provides sub-cent storage.")]

        retriever = RetrieverNode(retriever_fn=dummy_retriever, node_id="retriever")
        llm = LLMNode(model=MockChatModel(), node_id="llm")
        formatter = format_summary(
            node_id="formatter", config={"header": "Storage Analysis"}
        )

        graph = SimpleGraph()
        graph.add_node(retriever)
        graph.add_node(llm)
        graph.add_node(formatter)

        # Connect retriever.context -> llm.context
        graph.add_edge("retriever", "llm", source_port="context", target_port="context")
        # Connect llm.response -> formatter.response
        graph.add_edge("llm", "formatter", source_port="response", target_port="response")

        result = graph.run({
            "query": "How cheap is S3 Vectors?",
            "prompt": "Summarize vector cost",
        })

        assert "Storage Analysis" in result["report"]
        assert "Summarize vector cost" in result["report"]
        assert "S3 Vectors provides sub-cent storage" in result["report"]

        # Validate graph
        graph.validate()

        # Serialize & execute reloaded graph
        json_flow = graph.to_json()
        restored = SimpleGraph.from_json(json_flow)
        # Bind runtime execution handles to restored graph nodes
        restored_llm = restored.nodes["llm"]
        assert isinstance(restored_llm, LLMNode)
        restored_llm.model = MockChatModel()

        restored_retriever = restored.nodes["retriever"]
        assert isinstance(restored_retriever, RetrieverNode)
        restored_retriever.retriever_fn = dummy_retriever

        restored_res = restored.run({
            "query": "How cheap is S3 Vectors?",
            "prompt": "Summarize vector cost",
        })
        assert restored_res["report"] == result["report"]

    finally:
        node_registry.unregister(custom_type)


def test_custom_nodes_sdk_example_runs_in_ci() -> None:
    # Execute the example script to ensure docs and examples never regress
    from examples.custom_nodes_sdk import main

    main()

