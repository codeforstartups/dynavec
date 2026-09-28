"""Tests for the OpenAI embedder retry behavior."""
from types import SimpleNamespace

import pytest

from dynavec.embeddings.openai import OpenAIEmbedder


class _FakeEmbeddings:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0
        self.requests = []

    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)

        if self.calls == 1:
            raise self.error

        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
            ]
        )


class _FakeClient:
    def __init__(self, error: Exception):
        self.embeddings = _FakeEmbeddings(error)


@pytest.mark.parametrize("dimensions", [None, 2])
def test_openai_embedder_retries_rate_limit_and_respects_retry_after(
    monkeypatch, dimensions
):
    class RateLimitError(Exception):
        status_code = 429

        def __init__(self):
            self.response = SimpleNamespace(
                headers={"retry-after": "5"},
            )

    error = RateLimitError()

    embedder, fake_client = _make_embedder(error)
    embedder._requested_dim = dimensions

    delays = []
    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    result = embedder.embed_documents(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert fake_client.embeddings.calls == 2
    assert delays == [5.0]
    expected_request = {
        "model": "text-embedding-3-small",
        "input": ["hello"],
    }
    if dimensions is not None:
        expected_request["dimensions"] = dimensions
    assert fake_client.embeddings.requests == [expected_request, expected_request]

def test_openai_embedder_retries_server_error(monkeypatch):
    class ServerError(Exception):
        status_code = 500

    error = ServerError("internal server error")

    embedder, fake_client = _make_embedder(error)

    delays = []
    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    result = embedder.embed_documents(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert fake_client.embeddings.calls == 2
    assert len(delays) == 1

def test_openai_embedder_does_not_retry_client_error(monkeypatch):
    class ClientError(Exception):
        status_code = 400

    error = ClientError("bad request")

    embedder, fake_client = _make_embedder(error)
    embedder._client = fake_client

    delays = []
    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    with pytest.raises(ClientError):
        embedder.embed_documents(["hello"])

    assert fake_client.embeddings.calls == 1
    assert delays == []

def _make_embedder(error):
    fake_client = _FakeClient(error)

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder._client = fake_client
    embedder.model = "text-embedding-3-small"
    embedder._requested_dim = None
    embedder.dimension = 1536
    embedder.batch_size = 256

    return embedder, fake_client
