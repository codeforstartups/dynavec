"""Unit tests for dynaflow code <-> visual round-trip (import/export graph as code)."""

from __future__ import annotations

import ast
import sys
from typing import Any

import pytest

from dynavec import (
    GRAPH_SCHEMA_VERSION,
    ApprovalNode,
    CodeNode,
    LLMNode,
    RetrieverNode,
    RetryPolicy,
    SimpleGraph,
    ToolNode,
    custom_node,
    export_graph_to_code,
    import_graph_from_code,
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
        import json

        last = messages[-1].content
        return ChatResult(
            message=Message(
                role="assistant",
                content=json.dumps({"operation": "multiply", "factor": 3, "query": last}),
            ),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        yield from []


class MockDoc:
    def __init__(self, doc_id: str, text: str) -> None:
        self.id = doc_id
        self.text = text


# =====================================================================
# 1. Multi-Node Pipeline Export & Safe AST Import Tests
# =====================================================================


def test_multinode_pipeline_export_to_code() -> None:
    """Test exporting a multi-node pipeline (RetrieverNode -> LLMNode -> ToolNode) to Python code."""
    graph = SimpleGraph()

    retriever = RetrieverNode(
        node_id="retriever_1",
        config={"top_k": 5, "namespace": "kb_docs"},
    )
    llm = LLMNode(
        node_id="llm_1",
        config={"temperature": 0.2, "system_prompt": "You are a helpful analyst."},
    )
    tool = ToolNode(
        node_id="tool_1",
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.1, max_delay=1.0, jitter=False),
    )

    graph.add_node(retriever)
    graph.add_node(llm)
    graph.add_node(tool)

    graph.add_edge(
        source_id="retriever_1",
        target_id="llm_1",
        source_port="context",
        target_port="context",
    )
    graph.add_edge(
        source_id="llm_1",
        target_id="tool_1",
        source_port="response",
        target_port="arguments",
    )

    # Export via both standalone function and convenience method
    code_str = graph.to_code()
    assert code_str == export_graph_to_code(graph)

    # Verify code contents
    assert "from dynavec import" in code_str
    assert "SimpleGraph" in code_str
    assert "RetrieverNode" in code_str
    assert "LLMNode" in code_str
    assert "ToolNode" in code_str
    assert "RetryPolicy" in code_str
    assert 'node_id="retriever_1"' in code_str
    assert '"top_k": 5' in code_str
    assert '"namespace": \'kb_docs\'' in code_str or '"namespace": "kb_docs"' in code_str
    assert 'max_attempts=4' in code_str
    assert 'jitter=False' in code_str
    assert 'graph.add_edge(' in code_str
    assert 'source_id="retriever_1"' in code_str
    assert 'target_id="llm_1"' in code_str
    assert 'source_port="context"' in code_str
    assert 'target_port="context"' in code_str

    # Statically parse with Python's ast to ensure valid syntax
    parsed_ast = ast.parse(code_str)
    assert isinstance(parsed_ast, ast.Module)


def test_import_graph_from_code_ast_safe() -> None:
    """Test importing code back into SimpleGraph using safe AST parsing without exec/eval."""
    code_str = """
from dynavec import SimpleGraph, RetrieverNode, LLMNode, ToolNode, RetryPolicy

graph = SimpleGraph()

retriever = RetrieverNode(
    node_id="retriever_1",
    config={"top_k": 5, "namespace": "kb_docs"},
)
graph.add_node(retriever)

llm = LLMNode(
    node_id="llm_1",
    config={"temperature": 0.2, "system_prompt": "You are a helpful analyst."},
)
graph.add_node(llm)

tool = ToolNode(
    node_id="tool_1",
    retry_policy=RetryPolicy(max_attempts=4, base_delay=0.1, max_delay=1.0, jitter=False),
)
graph.add_node(tool)

graph.add_edge(
    source_id="retriever_1",
    target_id="llm_1",
    source_port="context",
    target_port="context",
)
graph.add_edge(
    source_id="llm_1",
    target_id="tool_1",
    source_port="response",
    target_port="arguments",
)
"""
    restored = SimpleGraph.from_code(code_str)
    assert restored.to_dict() == import_graph_from_code(code_str).to_dict()

    assert len(restored.nodes) == 3
    assert set(restored.nodes.keys()) == {"retriever_1", "llm_1", "tool_1"}
    assert restored.nodes["retriever_1"].node_type == "retriever"
    assert restored.nodes["retriever_1"].config == {"top_k": 5, "namespace": "kb_docs"}
    assert restored.nodes["llm_1"].node_type == "llm"
    assert restored.nodes["llm_1"].config["temperature"] == 0.2
    assert restored.nodes["tool_1"].node_type == "tool"
    assert restored.nodes["tool_1"].retry_policy is not None
    assert restored.nodes["tool_1"].retry_policy.max_attempts == 4
    assert restored.nodes["tool_1"].retry_policy.jitter is False

    assert len(restored.edges) == 2
    edge1 = restored.edges[0]
    assert edge1.source_node == "retriever_1"
    assert edge1.target_node == "llm_1"
    assert edge1.source_port == "context"
    assert edge1.target_port == "context"

    edge2 = restored.edges[1]
    assert edge2.source_node == "llm_1"
    assert edge2.target_node == "tool_1"
    assert edge2.source_port == "response"
    assert edge2.target_port == "arguments"


# =====================================================================
# 2. Lossless Round-Trip Guarantee Tests
# =====================================================================


def test_lossless_roundtrip_graph_to_code_to_dict() -> None:
    """Validate lossless round-trip guarantee: graph -> to_code() -> from_code() -> to_dict() == graph.to_dict()."""
    graph = SimpleGraph()

    retriever = RetrieverNode(
        node_id="retriever_1",
        config={"top_k": 3, "namespace": "default"},
    )
    llm = LLMNode(
        node_id="llm_1",
        config={"temperature": 0.7, "model_name": "claude-3-5-sonnet"},
    )
    tool = ToolNode(
        node_id="tool_1",
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.05, max_delay=2.0, jitter=True),
    )

    graph.add_node(retriever)
    graph.add_node(llm)
    graph.add_node(tool)

    graph.add_edge(
        source_id="retriever_1",
        target_id="llm_1",
        source_port="context",
        target_port="context",
    )
    graph.add_edge(
        source_id="llm_1",
        target_id="tool_1",
        source_port="response",
        target_port="arguments",
    )

    # 1. Graph -> Code
    code = graph.to_code()

    # 2. Code -> Graph
    reconstructed = SimpleGraph.from_code(code)

    # 3. Verify exact dictionary equality
    orig_dict = graph.to_dict()
    reconstructed_dict = reconstructed.to_dict()

    assert orig_dict["version"] == GRAPH_SCHEMA_VERSION
    assert reconstructed_dict["version"] == orig_dict["version"]
    assert reconstructed_dict["execution_order"] == orig_dict["execution_order"]
    assert reconstructed_dict["nodes"] == orig_dict["nodes"]
    assert reconstructed_dict["edges"] == orig_dict["edges"]
    assert reconstructed_dict == orig_dict


def test_lossless_roundtrip_code_canvas_json_code() -> None:
    """Validate lossless round-trip guarantee: code -> canvas/JSON -> code."""
    graph = SimpleGraph()
    n1 = RetrieverNode(node_id="node_a", config={"top_k": 10})
    n2 = LLMNode(node_id="node_b", config={"max_tokens": 1000})
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge(
        source_id="node_a",
        target_id="node_b",
        source_port="context",
        target_port="context",
    )

    code1 = graph.to_code()

    # 1. code -> SimpleGraph
    g1 = SimpleGraph.from_code(code1)

    # 2. SimpleGraph -> Canvas/JSON
    canvas_json = g1.to_json()

    # 3. Canvas/JSON -> SimpleGraph
    g2 = SimpleGraph.from_json(canvas_json)

    # 4. SimpleGraph -> code
    code2 = g2.to_code()

    assert code2 == code1


# =====================================================================
# 3. Re-execution of Reconstructed Graphs
# =====================================================================


def test_reexecution_of_reconstructed_pipeline() -> None:
    """Verify that a reconstructed graph can be bound to runtime handles and executed."""
    graph = SimpleGraph()

    retriever = RetrieverNode(node_id="retriever_1", config={"top_k": 2})
    llm = LLMNode(node_id="llm_1")
    tool = ToolNode(node_id="tool_1")

    graph.add_node(retriever)
    graph.add_node(llm)
    graph.add_node(tool)

    graph.add_edge(
        source_id="retriever_1",
        target_id="llm_1",
        source_port="context",
        target_port="context",
    )
    graph.add_edge(
        source_id="llm_1",
        target_id="tool_1",
        source_port="response",
        target_port="arguments",
    )

    # Round-trip through code
    code = graph.to_code()
    restored = SimpleGraph.from_code(code)

    # Bind runtime execution handles
    def mock_retriever_fn(query: str, top_k: int = 5, filter: Any = None) -> list[MockDoc]:
        return [
            MockDoc("doc1", f"Found info about {query}"),
            MockDoc("doc2", "Additional context"),
        ]

    def mock_tool_fn(operation: str, factor: int, query: str) -> dict[str, Any]:
        return {"status": "computed", "answer": f"Processed {operation} with factor {factor} on {query}"}

    restored_retriever = restored.nodes["retriever_1"]
    assert isinstance(restored_retriever, RetrieverNode)
    restored_retriever.retriever_fn = mock_retriever_fn

    restored_llm = restored.nodes["llm_1"]
    assert isinstance(restored_llm, LLMNode)
    restored_llm.model = MockChatModel()

    restored_tool = restored.nodes["tool_1"]
    assert isinstance(restored_tool, ToolNode)
    restored_tool.tool = mock_tool_fn

    result = restored.run({"query": "vector databases", "prompt": "Analyze vector storage"})

    assert "documents" in result
    assert "Found info about vector databases" in result["context"]
    assert "response" in result
    assert result["result"] == {
        "status": "computed",
        "answer": "Processed multiply with factor 3 on Context:\nFound info about vector databases\n\nAdditional context\n\nUser Question:\nAnalyze vector storage",
    }


def test_pure_execution_without_runtime_handles() -> None:
    """Verify execution of reconstructed graph with nodes that require no external models."""
    graph = SimpleGraph()
    step1 = CodeNode(node_id="step1", config={"factor": 10})
    step2 = ApprovalNode(node_id="step2")

    graph.add_node(step1)
    graph.add_node(step2)
    graph.add_edge("step1", "step2")

    code = graph.to_code()
    restored = SimpleGraph.from_code(code)

    # CodeNode default lambda passes through outputs; ApprovalNode passes approved inputs
    res = restored.run({"request": 42, "approved": True})
    assert res["data"] == 42
    assert res["approved"] is True
    assert restored.nodes["step1"].status == "succeeded"
    assert restored.nodes["step2"].status == "succeeded"


# =====================================================================
# 4. Error Edges, Port Mapping, and Status Preservation
# =====================================================================


def test_error_edge_and_fallback_roundtrip() -> None:
    """Test exporting and importing graphs containing fallback error edges."""
    graph = SimpleGraph()
    primary = CodeNode(node_id="primary_worker", config={"mode": "risky"})
    fallback = CodeNode(node_id="fallback_worker", config={"mode": "safe"})

    graph.add_node(primary)
    graph.add_node(fallback)
    graph.add_error_edge("primary_worker", "fallback_worker")

    code = graph.to_code()
    assert "is_error_edge=True" in code

    restored = SimpleGraph.from_code(code)
    assert len(restored.edges) == 1
    assert restored.edges[0].is_error_edge is True
    assert restored.edges[0].source_port == "error"
    assert restored.edges[0].target_port == "error"

    assert restored.to_dict() == graph.to_dict()


def test_custom_ports_and_mapping_roundtrip() -> None:
    """Test round-trip with port renaming and mapping."""
    graph = SimpleGraph()
    n1 = RetrieverNode(node_id="r1")
    n2 = LLMNode(node_id="l1")

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge(
        source_id="r1",
        target_id="l1",
        source_port="context",
        target_port="context",
        mapping={"context": "prompt"},
    )

    code = graph.to_code()
    assert "mapping=" in code

    restored = SimpleGraph.from_code(code)
    assert restored.edges[0].mapping == {"context": "prompt"}
    assert restored.to_dict() == graph.to_dict()


def test_node_status_preservation() -> None:
    """Test that non-pending node statuses are preserved across export and import."""
    graph = SimpleGraph()
    n1 = CodeNode(node_id="finished_step")
    n1.status = "succeeded"
    n2 = CodeNode(node_id="waiting_step")

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge("finished_step", "waiting_step")

    code = graph.to_code()
    assert 'finished_step.status = "succeeded"' in code

    restored = SimpleGraph.from_code(code)
    assert restored.nodes["finished_step"].status == "succeeded"
    assert restored.nodes["waiting_step"].status == "pending"
    assert restored.to_dict() == graph.to_dict()


# =====================================================================
# 5. Custom Node Round-Trip Tests
# =====================================================================


def test_custom_node_roundtrip() -> None:
    """Test round-trip of a custom node registered via custom_node SDK."""
    test_type = "test_roundtrip_sentiment"

    @custom_node(
        node_type=test_type,
        label="Roundtrip Sentiment",
        inputs=[("text", "string")],
        outputs=[("score", "number")],
        config_schema={"threshold": {"type": "number", "default": 0.5}},
    )
    def my_custom_logic(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {"score": 0.9}

    try:
        graph = SimpleGraph()
        cls = node_registry.get(test_type)
        assert cls is not None
        custom_inst = cls(node_id="custom_1", config={"threshold": 0.75})
        graph.add_node(custom_inst)

        code = graph.to_code()
        assert test_type in code

        restored = SimpleGraph.from_code(code)
        assert "custom_1" in restored.nodes
        assert restored.nodes["custom_1"].node_type == test_type
        assert restored.nodes["custom_1"].config == {"threshold": 0.75}

        assert restored.to_dict() == graph.to_dict()
    finally:
        node_registry.unregister(test_type)


# =====================================================================
# 6. Safe AST Parsing Security Tests (No exec/eval)
# =====================================================================


def test_safe_ast_parsing_no_exec_of_malicious_code() -> None:
    """Verify that malicious Python code or arbitrary expressions are NOT executed."""
    # Attempt code injection that would create a sentinel attribute if executed
    sentinel_attr = "_test_hack_executed_sentinel"
    assert not hasattr(sys, sentinel_attr)

    malicious_code = f"""
import sys
setattr(sys, "{sentinel_attr}", True)

from dynavec import SimpleGraph, CodeNode

graph = SimpleGraph()
safe_node = CodeNode(node_id="safe_step")
graph.add_node(safe_node)
"""
    restored = SimpleGraph.from_code(malicious_code)

    # CRITICAL: sys was NOT modified, arbitrary code was NOT executed
    assert not hasattr(sys, sentinel_attr)
    assert len(restored.nodes) == 1
    assert "safe_step" in restored.nodes


def test_syntax_error_handling() -> None:
    """Verify that invalid syntax raises ValueError with descriptive error message."""
    invalid_code = "graph = SimpleGraph(broken_syntax {{"
    with pytest.raises(ValueError, match="Invalid Python syntax in graph code"):
        SimpleGraph.from_code(invalid_code)


def test_unknown_node_class_error() -> None:
    """Verify that importing an unregistered node class raises a descriptive ValueError."""
    code_with_unknown = """
from dynavec import SimpleGraph
graph = SimpleGraph()
x = NonExistentSpecialNode(node_id="x1")
graph.add_node(x)
"""
    with pytest.raises(ValueError, match="Unknown node type or class 'NonExistentSpecialNode'"):
        SimpleGraph.from_code(code_with_unknown)


# =====================================================================
# 7. Flexible Syntax Styles & Edge Cases
# =====================================================================


def test_flexible_syntax_styles() -> None:
    """Verify that various idiomatic Python SDK patterns are correctly parsed."""
    flexible_code = """
from dynavec import SimpleGraph, RetrieverNode, LLMNode, ToolNode

# 1. Wrapped in function
def build_pipeline():
    graph = SimpleGraph()

    # 2. Variable assignments and add_node
    r = RetrieverNode(node_id="r1", config={"top_k": 3})
    graph.add_node(r)

    # 3. Direct inline instantiation in add_node
    graph.add_node(LLMNode(node_id="l1", config={"temperature": 0.5}))

    # 4. Node object references in add_edge
    graph.add_edge(r, "l1")

    # 5. Method chaining
    t = ToolNode(node_id="t1")
    graph.add_node(t).add_edge("l1", "t1")

    return graph
"""
    restored = SimpleGraph.from_code(flexible_code)

    assert len(restored.nodes) == 3
    assert {"r1", "l1", "t1"}.issubset(set(restored.nodes.keys()))
    assert len(restored.edges) == 2
    assert restored.edges[0].source_node == "r1"
    assert restored.edges[0].target_node == "l1"
    assert restored.edges[1].source_node == "l1"
    assert restored.edges[1].target_node == "t1"


def test_empty_graph_roundtrip() -> None:
    """Test exporting and importing an empty SimpleGraph."""
    empty_graph = SimpleGraph()
    code = empty_graph.to_code()
    assert "SimpleGraph()" in code

    restored = SimpleGraph.from_code(code)
    assert len(restored.nodes) == 0
    assert len(restored.edges) == 0
    assert restored.to_dict() == empty_graph.to_dict()


def test_node_id_with_special_characters_roundtrip() -> None:
    """Test nodes with hyphenated IDs or special characters."""
    graph = SimpleGraph()
    n1 = CodeNode(node_id="step-1-fetch")
    n2 = CodeNode(node_id="step-2-process")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge("step-1-fetch", "step-2-process")

    code = graph.to_code()
    # Code should have sanitized variable names but preserved exact string node_ids
    assert 'node_id="step-1-fetch"' in code
    assert 'node_id="step-2-process"' in code
    assert 'source_id="step-1-fetch"' in code
    assert 'target_id="step-2-process"' in code

    restored = SimpleGraph.from_code(code)
    assert "step-1-fetch" in restored.nodes
    assert "step-2-process" in restored.nodes
    assert restored.edges[0].source_node == "step-1-fetch"
    assert restored.edges[0].target_node == "step-2-process"
    assert restored.to_dict() == graph.to_dict()


def test_variable_references_in_ast_import() -> None:
    """Verify AST parser resolves variables for config, retry_policy, and edge mapping."""
    code = """
from dynavec import SimpleGraph, LLMNode, RetrieverNode, RetryPolicy

graph = SimpleGraph()

llm_cfg = {"temperature": 0.8, "model": "gpt-4o"}
r_policy = RetryPolicy(max_attempts=5, base_delay=0.1)
edge_map = {"context": "input"}

r = RetrieverNode(node_id="retriever_var", retry_policy=r_policy)
l = LLMNode(node_id="llm_var", config=llm_cfg)

graph.add_node(r)
graph.add_node(l)
graph.add_edge(source_id="retriever_var", target_id="llm_var", mapping=edge_map)
"""
    restored = SimpleGraph.from_code(code)
    assert "llm_var" in restored.nodes
    assert restored.nodes["llm_var"].config == {"temperature": 0.8, "model": "gpt-4o"}
    assert restored.nodes["retriever_var"].retry_policy is not None
    assert restored.nodes["retriever_var"].retry_policy.max_attempts == 5
    assert len(restored.edges) == 1
    assert restored.edges[0].mapping == {"context": "input"}


def test_custom_node_registry_not_overwritten() -> None:
    """Verify importing code with @custom_node does not overwrite existing registry implementations."""
    from dynavec.nodes.registry import NodeRegistry

    test_reg = NodeRegistry()

    @custom_node(node_type="test_existing_node", registry=test_reg)
    def my_real_func(inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"real_output": 123}

    code = """
from dynavec import custom_node, SimpleGraph

@custom_node(node_type="test_existing_node")
def my_real_func(inputs, config=None):
    pass

graph = SimpleGraph()
n = my_real_func(node_id="custom_1")
graph.add_node(n)
"""
    restored = import_graph_from_code(code, registry=test_reg)
    assert "custom_1" in restored.nodes
    # Verify the registry still points to the real class and function
    custom_cls = test_reg.get("test_existing_node")
    assert custom_cls is not None
    node_inst = custom_cls(node_id="custom_1")
    exec_result = node_inst.execute({})
    assert exec_result == {"real_output": 123}


def test_custom_node_export_includes_reminder_comment() -> None:
    """Verify exporting a graph with custom nodes emits a helpful reminder comment."""
    @custom_node(node_type="exporter_custom_reminder_node")
    def reminder_fn(inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    custom_cls = node_registry.get("exporter_custom_reminder_node")
    assert custom_cls is not None
    graph = SimpleGraph()
    n = custom_cls(node_id="rem_1")
    graph.add_node(n)

    code = graph.to_code()
    assert "# Note: Custom node type 'exporter_custom_reminder_node' must be registered before executing this script." in code
    assert "graph._execution_order" not in code
