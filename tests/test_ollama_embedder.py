"""Unit tests for OllamaEmbedder."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from dynavec.embeddings import OllamaEmbedder


def _fake_urlopen_response(data: dict):
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    return resp


def test_import_lazy():
    from dynavec.embeddings import OllamaEmbedder as Exported
    from dynavec.embeddings.ollama import OllamaEmbedder as Direct

    assert Exported is Direct


def test_default_parameters(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    emb = OllamaEmbedder()
    assert emb.host == "http://localhost:11434"
    assert emb.model == "nomic-embed-text"
    assert emb.dimension == 768
    assert emb.batch_size == 64


def test_custom_host_and_model():
    emb = OllamaEmbedder(host="http://my-ollama:11434/", model="all-minilm")
    assert emb.host == "http://my-ollama:11434"
    assert emb.model == "all-minilm"
    assert emb.dimension == 384


def test_env_var_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama-env:11434")
    emb = OllamaEmbedder()
    assert emb.host == "http://ollama-env:11434"


@pytest.mark.parametrize(
    ("model", "expected_dim"),
    [
        ("nomic-embed-text", 768),
        ("all-minilm", 384),
        ("bge-m3", 1024),
        ("mxbai-embed-large", 1024),
        ("snowflake-arctic-embed", 1024),
        ("unknown-model", 768),
    ],
)
def test_model_dimensions(model, expected_dim):
    assert OllamaEmbedder(model=model).dimension == expected_dim


def test_explicit_dimension_override():
    emb = OllamaEmbedder(model="nomic-embed-text", dimension=512)
    assert emb.dimension == 512


@patch("urllib.request.urlopen")
def test_embed_documents(mock_urlopen):
    mock_urlopen.return_value = _fake_urlopen_response(
        {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    )
    emb = OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")
    res = emb.embed_documents(["hello", "world"])

    assert res == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://localhost:11434/api/embed"
    assert req.method == "POST"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload == {"model": "nomic-embed-text", "input": ["hello", "world"]}


@patch("urllib.request.urlopen")
def test_embed_query(mock_urlopen):
    mock_urlopen.return_value = _fake_urlopen_response({"embeddings": [[0.1, 0.2, 0.3]]})
    emb = OllamaEmbedder()
    res = emb.embed_query("query text")

    assert res == [0.1, 0.2, 0.3]


@patch("urllib.request.urlopen")
def test_batching(mock_urlopen):
    mock_urlopen.side_effect = [
        _fake_urlopen_response({"embeddings": [[0.1], [0.2]]}),
        _fake_urlopen_response({"embeddings": [[0.3]]}),
    ]
    emb = OllamaEmbedder(batch_size=2)
    res = emb.embed_documents(["a", "b", "c"])

    assert res == [[0.1], [0.2], [0.3]]
    assert mock_urlopen.call_count == 2


def test_empty_input():
    emb = OllamaEmbedder()
    assert emb.embed_documents([]) == []


@patch("urllib.request.urlopen")
def test_http_error_handling(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    emb = OllamaEmbedder()
    with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
        emb.embed_documents(["test"])
