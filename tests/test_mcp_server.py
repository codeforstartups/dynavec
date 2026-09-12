"""Unit tests for FastMCP server and environment-based client configuration."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from dynavec.cli import _parser
from dynavec.exceptions import ConfigurationError, MissingDependencyError
from dynavec.mcp.server import client_from_env, create_mcp_server
from dynavec.models import SearchResult


class _FakeTool:
    def __init__(self, fn, name=None, description=None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__


class _FakeFastMCP:
    """Mock FastMCP object capturing registered tools."""

    def __init__(self, name="dynavec", **kwargs):
        self.name = name
        self.tools = {}
        self.settings = types.SimpleNamespace(port=8000)

    def tool(self, name=None, description=None):
        def decorator(fn):
            tool_obj = _FakeTool(fn, name=name, description=description)
            self.tools[tool_obj.name] = tool_obj
            return fn

        return decorator

    def run(self, transport="stdio"):
        self.last_transport = transport


@pytest.fixture
def fake_mcp(monkeypatch):
    """Inject a fake mcp.server.fastmcp module into sys.modules."""
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _FakeFastMCP

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    return fastmcp_mod


def test_missing_mcp_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)

    with pytest.raises(MissingDependencyError) as exc:
        create_mcp_server()
    assert "pip install 'dynavec[mcp]'" in str(exc.value)


def test_client_from_env_missing_required():
    with pytest.raises(ConfigurationError, match="Missing required environment variable"):
        client_from_env({})

    with pytest.raises(ConfigurationError, match="DYNAVEC_INDEX"):
        client_from_env({"DYNAVEC_VECTOR_BUCKET": "b", "DYNAVEC_TABLE": "t"})


def test_client_from_env_full_configuration(monkeypatch):
    import dynavec.embeddings.openai as oai_mod

    mock_oai = MagicMock()
    mock_oai.dimension = 1536
    monkeypatch.setattr(oai_mod, "OpenAIEmbedder", lambda **kw: mock_oai)

    env = {
        "DYNAVEC_VECTOR_BUCKET": "my-bucket",
        "DYNAVEC_INDEX": "my-index",
        "DYNAVEC_TABLE": "my-table",
        "DYNAVEC_DIMENSION": "1536",
        "DYNAVEC_REGION": "us-west-2",
        "DYNAVEC_FILTERABLE_KEYS": "topic, category",
        "DYNAVEC_DISTANCE_METRIC": "euclidean",
        "OPENAI_API_KEY": "sk-test",
    }
    client = client_from_env(env)
    assert client.config.vector_bucket == "my-bucket"
    assert client.config.index == "my-index"
    assert client.config.table == "my-table"
    assert client.config.dimension == 1536
    assert client.config.region == "us-west-2"
    assert client.config.filterable_keys == ["topic", "category"]
    assert client.config.distance_metric == "euclidean"
    assert client.embedder is mock_oai


def test_client_from_env_embedder_resolution(monkeypatch):
    import dynavec.embeddings.gemini as gemini_mod
    import dynavec.embeddings.mistral as mistral_mod
    import dynavec.embeddings.sentence_transformers as st_mod
    import dynavec.embeddings.voyage as voyage_mod

    monkeypatch.setattr(gemini_mod, "GeminiEmbedder", lambda **kw: MagicMock(dimension=768))
    monkeypatch.setattr(voyage_mod, "VoyageEmbedder", lambda **kw: MagicMock(dimension=1024))
    monkeypatch.setattr(mistral_mod, "MistralEmbedder", lambda **kw: MagicMock(dimension=1024))
    monkeypatch.setattr(st_mod, "SentenceTransformerEmbedder", lambda **kw: MagicMock(dimension=384))

    base_env = {
        "DYNAVEC_BUCKET": "b",
        "DYNAVEC_INDEX": "i",
        "DYNAVEC_TABLE": "t",
        "DYNAVEC_REGION": "us-east-1",
    }

    # Gemini
    c1 = client_from_env({**base_env, "GEMINI_API_KEY": "gkey"})
    assert c1.embedder is not None
    assert c1.config.dimension == 768

    # Voyage
    c2 = client_from_env({**base_env, "VOYAGE_API_KEY": "vkey"})
    assert c2.embedder is not None
    assert c2.config.dimension == 1024

    # Mistral
    c3 = client_from_env({**base_env, "MISTRAL_API_KEY": "mkey"})
    assert c3.embedder is not None
    assert c3.config.dimension == 1024

    # Sentence Transformers
    c4 = client_from_env({**base_env, "DYNAVEC_EMBEDDER": "sentence-transformers"})
    assert c4.embedder is not None
    assert c4.config.dimension == 384

    # None
    c5 = client_from_env({**base_env, "DYNAVEC_EMBEDDER": "none"})
    assert c5.embedder is None
    assert c5.config.dimension == 1536


def test_mcp_server_search_tool(fake_mcp):
    mock_db = MagicMock()
    mock_db.search.return_value = [
        SearchResult(id="doc-1", score=0.95, text="Hello world", metadata={"topic": "greeting"}),
        SearchResult(id="doc-2", score=0.82, text="How are you", metadata={}),
    ]

    mcp = create_mcp_server(mock_db, name="test-dynavec")
    assert "dynavec_search" in mcp.tools
    assert "dynavec_graph_search" in mcp.tools

    search_fn = mcp.tools["dynavec_search"].fn
    result_text = search_fn(
        query="test query",
        top_k=2,
        namespace="custom-ns",
        filter_json=json.dumps({"topic": "greeting"}),
        rescore="dot",
        rerank="mmr",
    )

    mock_db.search.assert_called_once_with(
        query="test query",
        top_k=2,
        namespace="custom-ns",
        filter={"topic": "greeting"},
        rescore="dot",
        rerank="mmr",
    )
    assert "[1] id: doc-1 (score: 0.9500 | metadata: {\"topic\": \"greeting\"})" in result_text
    assert "Hello world" in result_text
    assert "[2] id: doc-2 (score: 0.8200)" in result_text


def test_mcp_server_search_tool_empty_and_error(fake_mcp):
    mock_db = MagicMock()
    mock_db.search.return_value = []

    mcp = create_mcp_server(mock_db)
    search_fn = mcp.tools["dynavec_search"].fn

    empty_res = search_fn(query="missing", namespace="kb")
    assert "No results found for query 'missing' in namespace 'kb'." in empty_res

    err_res = search_fn(query="q", filter_json="{invalid json")
    assert "Error parsing filter_json" in err_res


def test_mcp_server_graph_search_tool(fake_mcp):
    mock_db = MagicMock()
    mock_db.graph_search.return_value = [
        SearchResult(id="doc-graph", score=0.91, text="Graph entity content", metadata={"entity": "acme"}),
    ]

    mcp = create_mcp_server(mock_db)
    graph_fn = mcp.tools["dynavec_graph_search"].fn

    result_text = graph_fn(
        query="company news",
        seed_entities=["acme", "globex"],
        hops=3,
        top_k=5,
        namespace="kb",
    )

    mock_db.graph_search.assert_called_once_with(
        query="company news",
        seed_entities=["acme", "globex"],
        hops=3,
        top_k=5,
        namespace="kb",
    )
    assert "Found 1 results for query 'company news'" in result_text
    assert "id: doc-graph" in result_text
    assert "Graph entity content" in result_text


def test_cli_mcp_subcommand_parser():
    parser = _parser()
    args = parser.parse_args(["mcp", "--transport", "sse", "--port", "9000"])
    assert args.command == "mcp"
    assert args.transport == "sse"
    assert args.port == 9000
