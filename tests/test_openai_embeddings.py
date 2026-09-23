"""Tests for the OpenAI embedder retry behavior."""
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from dynavec.embeddings.openai import OpenAIEmbedder


class _FakeEmbeddings:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1

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


def test_openai_embedder_retries_rate_limit_and_respects_retry_after(monkeypatch):
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/embeddings",
    )

    response = httpx.Response(
        429,
        headers={"Retry-After": "5"},
        request=request,
    )

    error = RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )

    embedder = OpenAIEmbedder(api_key="test-key")
    fake_client = _FakeClient(error)
    embedder._client = fake_client

    delays = []
    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    result = embedder.embed_documents(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert fake_client.embeddings.calls == 2
    assert delays == [5.0]

def test_openai_embedder_retries_server_error(monkeypatch):
    class ServerError(Exception):
        status_code = 500

    error = ServerError("internal server error")

    embedder = OpenAIEmbedder(api_key="test-key")
    fake_client = _FakeClient(error)
    embedder._client = fake_client

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

    embedder = OpenAIEmbedder(api_key="test-key")
    fake_client = _FakeClient(error)
    embedder._client = fake_client

    delays = []
    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    with pytest.raises(ClientError):
        embedder.embed_documents(["hello"])

    assert fake_client.embeddings.calls == 1
    assert delays == []
