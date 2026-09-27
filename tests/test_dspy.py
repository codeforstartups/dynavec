"""Tests for the DSPy retrieval integration."""

import pytest

dspy = pytest.importorskip("dspy")

from dynavec.integrations.dspy import DynavecRM  # noqa: E402
from dynavec.models import SearchResult  # noqa: E402


class _FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            SearchResult(
                id="doc-1",
                score=0.95,
                text="Retrieval-augmented generation combines retrieval with generation.",
                metadata={"topic": "rag"},
            ),
            SearchResult(
                id="doc-2",
                score=0.88,
                text="Dynavec provides vector retrieval backed by AWS.",
                metadata={"topic": "dynavec"},
            ),
        ]


def test_dspy_retrieve_uses_dynavec():
    client = _FakeClient()
    rm = DynavecRM(client, namespace="kb", k=3)

    previous_rm = dspy.settings.rm
    try:
        dspy.configure(rm=rm)

        retriever = dspy.Retrieve(k=2)
        result = retriever(
            "what is retrieval-augmented generation?",
            filter={"topic": "rag"},
        )
    finally:
        dspy.configure(rm=previous_rm)

    assert result.passages == [
        "Retrieval-augmented generation combines retrieval with generation.",
        "Dynavec provides vector retrieval backed by AWS.",
    ]

    assert client.calls == [
        (
            "what is retrieval-augmented generation?",
            {
                "top_k": 2,
                "namespace": "kb",
                "filter": {"topic": "rag"},
            },
        )
    ]


def test_dynavec_rm_returns_dspy_passages():
    client = _FakeClient()
    rm = DynavecRM(client, namespace="kb", k=3)

    passages = rm("dynavec")

    assert len(passages) == 2
    assert isinstance(passages[0], dspy.Prediction)

    assert passages[0].long_text == (
        "Retrieval-augmented generation combines retrieval with generation."
    )
    assert passages[0].id == "doc-1"
    assert passages[0].score == 0.95
    assert passages[0].metadata == {"topic": "rag"}
    assert passages[0]["id"] == "doc-1"

    assert client.calls == [
        (
            "dynavec",
            {
                "top_k": 3,
                "namespace": "kb",
            },
        )
    ]


def test_dynavec_rm_allows_k_override_and_search_kwargs():
    client = _FakeClient()
    rm = DynavecRM(client, namespace="docs", k=3)

    rm(
        "vector search",
        k=5,
        rerank="mmr",
        mmr_lambda=0.7,
    )

    assert client.calls == [
        (
            "vector search",
            {
                "top_k": 5,
                "namespace": "docs",
                "rerank": "mmr",
                "mmr_lambda": 0.7,
            },
        )
    ]
