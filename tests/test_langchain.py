"""Tests for the LangChain vector store integration."""

from dynavec.integrations.langchain import DynavecVectorStore
from dynavec.models import SearchResult


class _FakeClient:
    embedder = None

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            SearchResult(
                id="doc-1",
                score=0.95,
                text="Async Dynavec result",
                metadata={"topic": "ai"},
            )
        ]


def _fail_sync(*args, **kwargs):
    raise AssertionError("sync fallback should not be used")


async def test_async_retriever_ainvoke(monkeypatch):
    client = _FakeClient()
    store = DynavecVectorStore(client, namespace="kb")

    monkeypatch.setattr(store, "similarity_search", _fail_sync)

    retriever = store.as_retriever(search_kwargs={"k": 4})
    docs = await retriever.ainvoke("machine learning")

    assert len(docs) == 1
    assert docs[0].page_content == "Async Dynavec result"
    assert docs[0].metadata == {
        "topic": "ai",
        "id": "doc-1",
        "score": 0.95,
    }

    assert client.calls == [
        (
            "machine learning",
            {
                "top_k": 4,
                "namespace": "kb",
                "filter": None,
            },
        )
    ]


async def test_async_similarity_search_with_score(monkeypatch):
    client = _FakeClient()
    store = DynavecVectorStore(client, namespace="kb")

    monkeypatch.setattr(store, "similarity_search_with_score", _fail_sync)

    results = await store.asimilarity_search_with_score(
        "machine learning",
        k=2,
        filter={"topic": "ai"},
    )

    assert len(results) == 1

    document, score = results[0]

    assert document.page_content == "Async Dynavec result"
    assert document.metadata == {
        "topic": "ai",
        "id": "doc-1",
    }
    assert score == 0.95

    assert client.calls == [
        (
            "machine learning",
            {
                "top_k": 2,
                "namespace": "kb",
                "filter": {"topic": "ai"},
            },
        )
    ]


async def test_async_max_marginal_relevance_search(monkeypatch):
    client = _FakeClient()
    store = DynavecVectorStore(client, namespace="kb")

    monkeypatch.setattr(store, "max_marginal_relevance_search", _fail_sync)

    docs = await store.amax_marginal_relevance_search(
        "machine learning",
        k=3,
        lambda_mult=0.7,
        filter={"topic": "ai"},
    )

    assert len(docs) == 1
    assert docs[0].page_content == "Async Dynavec result"
    assert docs[0].metadata == {
        "topic": "ai",
        "id": "doc-1",
        "score": 0.95,
    }

    assert client.calls == [
        (
            "machine learning",
            {
                "top_k": 3,
                "namespace": "kb",
                "filter": {"topic": "ai"},
                "rerank": "mmr",
                "mmr_lambda": 0.7,
            },
        )
    ]
