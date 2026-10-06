"""Unit tests for dynaflow built-in node library and graph execution pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from dynavec import (
    ApprovalNode,
    CodeNode,
    LLMNode,
    LoopNode,
    NodePort,
    NodeSchema,
    RetrieverNode,
    RouterNode,
    SimpleGraph,
    SubGraphNode,
    ToolNode,
    WebhookNode,
    node_registry,
)
from dynavec.chat.base import ChatModel, ChatResult, Message


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
    ):
        last = messages[-1].content
        from dynavec.chat.base import ChatChunk

        yield ChatChunk(content=f"Processed: {last}")


class MockDoc:
    def __init__(self, doc_id: str, text: str, score: float = 0.9):
        self.id = doc_id
        self.text = text
        self.score = score
        self.metadata = {"source": "unit-test"}


# =====================================================================
# Registry & Schema Tests
# =====================================================================


def test_registry_registration_and_palette():
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


def test_port_and_schema_serialization():
    port = NodePort(
        name="test_in",
        port_type="string",
        description="test input port",
        required=True,
        default="abc",
    )
    port_dict = port.to_dict()
    assert port_dict["name"] == "test_in"
    assert port_dict["port_type"] == "string"
    assert port_dict["required"] is True
    assert port_dict["default"] == "abc"

    schema = NodeSchema(
        node_type="custom_test",
        label="Custom Test",
        description="A test node schema",
        inputs=[port],
        outputs=[],
    )
    s_dict = schema.to_dict()
    assert s_dict["node_type"] == "custom_test"
    assert len(s_dict["inputs"]) == 1

    rf_dict = schema.to_reactflow_dict()
    assert rf_dict["type"] == "custom_test"
    assert rf_dict["inputPorts"] == ["test_in"]


# =====================================================================
# Individual Built-in Node Tests
# =====================================================================


def test_retriever_node():
    def dummy_retriever(query: str, top_k: int = 5, filter: dict[str, Any] | None = None):
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


def test_llm_node_chat_model():
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


def test_llm_node_callable():
    def simple_callable(messages: list[Message]) -> str:
        return f"Echo: {messages[-1].content}"

    node = LLMNode(model=simple_callable)
    outputs = node.execute({"prompt": "Hello world"})
    assert outputs["response"] == "Echo: Hello world"


def test_tool_node_success():
    def add(a: int, b: int) -> int:
        return a + b

    node = ToolNode(tool=add)
    outputs = node.execute({"arguments": {"a": 10, "b": 25}})
    assert outputs["status"] == "success"
    assert outputs["result"] == 35


def test_tool_node_json_string_and_error():
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


def test_router_node():
    node = RouterNode(route_fn=lambda x: "admin_flow" if x.get("is_admin") else "user_flow")

    out1 = node.execute({"data": {"is_admin": True}})
    assert out1["selected_route"] == "admin_flow"
    assert out1["data"] == {"is_admin": True}

    out2 = node.execute({"data": {"is_admin": False}})
    assert out2["selected_route"] == "user_flow"

    # Explicit route override
    out3 = node.execute({"data": {}, "route": "manual_override"})
    assert out3["selected_route"] == "manual_override"


def test_code_node():
    def custom_transform(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"uppercased": inputs["text"].upper(), "len": len(inputs["text"])}

    node = CodeNode(code_fn=custom_transform)
    res = node.execute({"inputs": {"text": "dynaflow"}})
    assert res["uppercased"] == "DYNAFLOW"
    assert res["len"] == 8

    # When code returns non-dict
    scalar_node = CodeNode(code_fn=lambda inp: len(inp["text"]))
    res2 = scalar_node.execute({"inputs": {"text": "test"}})
    assert res2["outputs"] == 4


def test_approval_node():
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


def test_loop_node():
    node = LoopNode(map_fn=lambda x: x * 2, config={"max_iterations": 3})
    out = node.execute({"items": [1, 2, 3, 4, 5]})
    assert out["items"] == [2, 4, 6]  # clipped to max 3
    assert out["count"] == 3


def test_webhook_node():
    node = WebhookNode()
    payload = {"event": "user.signup", "user_id": "usr_123"}
    out = node.execute({"payload": payload, "headers": {"Authorization": "Bearer tok"}})
    assert out["payload"] == payload


def test_subgraph_node():
    def nested_workflow(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"processed_by_subgraph": True, "value": inputs.get("val", 0) + 10}

    node = SubGraphNode(subgraph=nested_workflow)
    out = node.execute({"inputs": {"val": 5}})
    assert out["outputs"]["processed_by_subgraph"] is True
    assert out["outputs"]["value"] == 15


# =====================================================================
# SimpleGraph Pipeline Integration: Retriever -> LLM -> Tool
# =====================================================================


def test_simple_graph_pipeline_retriever_llm_tool():
    # 1. RetrieverNode: searches knowledge base
    def mock_retriever(query: str, top_k: int = 5, filter: dict[str, Any] | None = None):
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


def test_simple_graph_validation_errors():
    graph = SimpleGraph()
    with pytest.raises(ValueError, match="Source node 'nonexistent' not in graph"):
        graph.add_edge("nonexistent", "target")

    node = CodeNode(code_fn=lambda x: x, node_id="valid_node")
    graph.add_node(node)
    with pytest.raises(ValueError, match="Target node 'missing_target' not in graph"):
        graph.add_edge("valid_node", "missing_target")
