"""Tests for the OpenAI Assistant and Function Calling tool adapter.

These tests run 100% offline with zero external network or API key dependencies.
"""

from types import SimpleNamespace

from dynavec.integrations.tools import OpenAIAssistantTool, as_openai_tool
from dynavec.models import SearchResult
from dynavec.namespace import NamespaceView


class _FakeClient:
    """Stands in for Dynavec client .search()."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            SearchResult(id="doc-1", score=0.92, text="Dynavec stores vectors on DynamoDB & S3."),
            SearchResult(id="doc-2", score=0.85, text="Serverless hybrid search is fast."),
        ]


def test_schema_structure_default():
    client = _FakeClient()
    tool = as_openai_tool(client)

    schema = tool.schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dynavec_search"
    assert "Search the knowledge base" in schema["function"]["description"]

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["required"] == ["query"]
    assert params["additionalProperties"] is False

    # Check aliases
    assert tool.to_openai_tool() == schema
    assert tool.to_dict() == schema


def test_schema_structure_custom_name_and_desc():
    client = _FakeClient()
    tool = as_openai_tool(
        client,
        name="custom_search",
        description="Search customer knowledge base",
    )

    schema = tool.schema
    assert schema["function"]["name"] == "custom_search"
    assert schema["function"]["description"] == "Search customer knowledge base"


def test_direct_callable_execution():
    client = _FakeClient()
    tool = as_openai_tool(client, top_k=2, namespace="kb")

    output = tool("vector indexing")
    assert "Dynavec stores vectors" in output
    assert "Serverless hybrid search" in output
    assert client.calls == [
        (
            "vector indexing",
            {"top_k": 2, "namespace": "kb", "filter": None, "rescore": None},
        )
    ]


def test_handle_tool_call_object_sdk_style():
    client = _FakeClient()
    tool = as_openai_tool(client)

    # Mimics openai.types.beta.threads.runs.RequiredActionFunctionToolCall
    call_obj = SimpleNamespace(
        id="call_abc123",
        type="function",
        function=SimpleNamespace(
            name="dynavec_search",
            arguments='{"query": "dynamodb storage"}',
        ),
    )

    output = tool.handle_tool_call(call_obj)
    assert output["tool_call_id"] == "call_abc123"
    assert "Dynavec stores vectors" in output["output"]


def test_handle_tool_call_dict_format():
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_xyz789",
        "type": "function",
        "function": {
            "name": "dynavec_search",
            "arguments": '{"query": "pricing model"}',
        },
    }

    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_xyz789"
    assert "Dynavec stores vectors" in output["output"]


def test_handle_tool_call_dict_arguments():
    """Some orchestrators (LiteLLM, Autogen) pass pre-parsed dicts in arguments."""
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_parsed",
        "function": {
            "name": "dynavec_search",
            "arguments": {"query": "preparsed query"},
        },
    }

    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_parsed"
    assert "Dynavec stores vectors" in output["output"]


def test_handle_tool_call_fallback_argument_keys():
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_alt",
        "function": {
            "name": "dynavec_search",
            "arguments": '{"search_query": "alternative key test"}',
        },
    }

    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_alt"
    assert client.calls[-1][0] == "alternative key test"


def test_handle_tool_call_malformed_json_graceful():
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_bad_json",
        "function": {
            "name": "dynavec_search",
            "arguments": "unquoted raw string query",
        },
    }

    # Should gracefully use the raw string rather than crashing
    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_bad_json"
    assert client.calls[-1][0] == "unquoted raw string query"


def test_handle_tool_call_retriever_exception_safe():
    def failing_retriever(query: str) -> str:
        raise RuntimeError("Database connection timeout")

    tool = OpenAIAssistantTool(failing_retriever, name="dynavec_search")
    call_dict = {
        "id": "call_err",
        "function": {"name": "dynavec_search", "arguments": '{"query": "error"}'},
    }

    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_err"
    assert "Error executing tool 'dynavec_search': Database connection timeout" in output["output"]


def test_submit_tool_outputs_batch():
    client = _FakeClient()
    tool = as_openai_tool(client, name="dynavec_search")

    tool_calls = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="dynavec_search", arguments='{"query": "first query"}'),
        ),
        SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(name="dynavec_search", arguments='{"query": "second query"}'),
        ),
    ]

    outputs = tool.submit_tool_outputs(tool_calls)
    assert len(outputs) == 2
    assert outputs[0]["tool_call_id"] == "call_1"
    assert outputs[1]["tool_call_id"] == "call_2"


def test_submit_tool_outputs_filters_foreign_tools():
    client = _FakeClient()
    tool = as_openai_tool(client, name="dynavec_search")

    tool_calls = [
        {"id": "call_weather", "function": {"name": "get_weather", "arguments": "{}"}},
        {
            "id": "call_dynavec",
            "function": {"name": "dynavec_search", "arguments": '{"query": "architecture"}'},
        },
        {"id": "call_calc", "function": {"name": "calculator", "arguments": "{}"}},
    ]

    outputs = tool.submit_tool_outputs(tool_calls)
    assert len(outputs) == 1
    assert outputs[0]["tool_call_id"] == "call_dynavec"


def test_namespace_view_support():
    client = _FakeClient()
    ns_view = NamespaceView(client, "billing")
    tool = as_openai_tool(ns_view, top_k=3, include_scores=True)

    output = tool("invoice questions")
    assert "[0.920] Dynavec stores vectors on DynamoDB & S3." in output
    assert client.calls == [
        (
            "invoice questions",
            {"top_k": 3, "namespace": "billing", "filter": None, "rescore": None},
        )
    ]


def test_handle_tool_call_null_and_none_arguments():
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_null",
        "function": {"name": "dynavec_search", "arguments": "null"},
    }
    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_null"
    assert client.calls[-1][0] == ""


def test_handle_tool_call_list_arguments():
    client = _FakeClient()
    tool = as_openai_tool(client)

    call_dict = {
        "id": "call_list",
        "function": {"name": "dynavec_search", "arguments": '["serverless", "database"]'},
    }
    output = tool.handle_tool_call(call_dict)
    assert output["tool_call_id"] == "call_list"
    assert client.calls[-1][0] == "serverless database"


def test_submit_tool_outputs_none_and_empty():
    client = _FakeClient()
    tool = as_openai_tool(client)

    assert tool.submit_tool_outputs(None) == []
    assert tool.submit_tool_outputs([]) == []


def test_as_openai_tool_forwarding_kwargs():
    client = _FakeClient()
    rescore_marker = object()
    tool = as_openai_tool(
        client,
        top_k=7,
        namespace="custom_ns",
        filter={"type": "pdf"},
        rescore=rescore_marker,
        join=" --- ",
        include_scores=True,
    )

    output = tool("search test")
    assert " --- " in output
    assert "[0.920]" in output
    assert client.calls[-1] == (
        "search test",
        {
            "top_k": 7,
            "namespace": "custom_ns",
            "filter": {"type": "pdf"},
            "rescore": rescore_marker,
        },
    )
