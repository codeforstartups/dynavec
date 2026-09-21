"""Tests for the Haystack integration."""
# ruff: noqa: E402
import pytest

pytest.importorskip("haystack", reason="haystack-ai not installed; pip install 'dynavec[haystack]'")

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from dynavec.config import DynavecConfig
from dynavec.integrations.haystack import DynavecDocumentStore, DynavecRetriever
from dynavec.models import SearchResult


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.config = DynavecConfig(
            vector_bucket="test-bucket",
            index="test-index",
            table="test-table",
            dimension=3,
            region="us-east-1",
        )

    def list_vectors(self, **kwargs):
        return iter([])

    def get(self, ids, **kwargs):
        self.calls.append(("get", ids, kwargs))
        return []

    def upsert(self, documents, **kwargs):
        self.calls.append(("upsert", documents, kwargs))


def test_write_documents():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    document = Document(
        id="doc-1",
        content="How do I reset my password?",
        meta={"category": "account"},
    )

    count = store.write_documents([document])

    assert count == 1

    operation, documents, kwargs = client.calls[-1]

    assert operation == "upsert"
    assert len(documents) == 1

    dynavec_document = documents[0]

    assert dynavec_document.id == "doc-1"
    assert dynavec_document.text == "How do I reset my password?"
    assert dynavec_document.metadata == {"category": "account"}
    assert dynavec_document.vector is None

    assert kwargs == {"namespace": "kb"}


def test_write_documents_with_embedding():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    document = Document(
        id="doc-2",
        content="Password reset instructions",
        meta={"category": "account"},
        embedding=[0.1, 0.2, 0.3],
    )

    count = store.write_documents([document])

    assert count == 1

    dynavec_document = client.calls[-1][1][0]

    assert dynavec_document.id == "doc-2"
    assert dynavec_document.text == "Password reset instructions"
    assert dynavec_document.metadata == {"category": "account"}
    assert dynavec_document.vector == [0.1, 0.2, 0.3]

def test_write_documents_overwrite_existing():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    document = Document(
        id="doc-1",
        content="Updated password instructions",
        meta={"category": "account"},
    )

    client.get = lambda ids, **kwargs: [
        type("Result", (), {"id": "doc-1"})()
    ]

    count = store.write_documents(
        [document],
        policy=DuplicatePolicy.OVERWRITE,
    )

    assert count == 1

    operation, documents, kwargs = client.calls[-1]

    assert operation == "upsert"
    assert documents[0].id == "doc-1"
    assert documents[0].text == "Updated password instructions"
    assert kwargs == {"namespace": "kb"}

def test_filter_documents():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    client.list_vectors = lambda **kwargs: iter([
        type(
            "Result",
            (),
            {
                "id": "doc-1",
                "text": "Password reset",
                "metadata": {"category": "account"},
                "vector": None,
            },
        )(),
        type(
            "Result",
            (),
            {
                "id": "doc-2",
                "text": "Home loan details",
                "metadata": {"category": "loan"},
                "vector": None,
            },
        )(),
    ])

    documents = store.filter_documents(
        filters={"category": "account"}
    )

    assert len(documents) == 1
    assert documents[0].id == "doc-1"
    assert documents[0].content == "Password reset"
    assert documents[0].meta == {"category": "account"}

def test_retriever():
    client = _FakeClient()

    client.search = lambda **kwargs: [
        SearchResult(
            id="doc-1",
            score=0.95,
            text="Password reset instructions",
            metadata={"category": "support"},
        )
    ]

    retriever = DynavecRetriever(client)

    result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

    assert len(result["documents"]) == 1
    assert result["documents"][0].id == "doc-1"
    assert result["documents"][0].content == "Password reset instructions"

def test_retriever_passes_top_k_and_filters():
    client = _FakeClient()
    client.search_calls = []

    def search(**kwargs):
        client.search_calls.append(kwargs)
        return []

    client.search = search

    retriever = DynavecRetriever(
        client,
        namespace="support",
        top_k=5,
    )

    retriever.run(
        query_embedding=[0.1, 0.2, 0.3],
        filters={"category": "support"},
    )

    assert len(client.search_calls) == 1

    call = client.search_calls[0]

    assert call["vector"] == [0.1, 0.2, 0.3]
    assert call["top_k"] == 5
    assert call["namespace"] == "support"
    assert call["filter"] == {"category": "support"}

def test_retriever_overrides_default_top_k():
    client = _FakeClient()
    client.search_calls = []

    def search(**kwargs):
        client.search_calls.append(kwargs)
        return []

    client.search = search

    retriever = DynavecRetriever(client, top_k=10)

    retriever.run(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
    )

    assert client.search_calls[0]["top_k"] == 3

def test_retriever_with_filters():
    client = _FakeClient()
    client.search_calls = []

    def search(**kwargs):
        client.search_calls.append(kwargs)
        return [
            SearchResult(
                id="doc-1",
                score=0.9,
                text="Reset password",
                metadata={"category": "support"},
            )
        ]

    client.search = search

    retriever = DynavecRetriever(client)

    result = retriever.run(
        query_embedding=[0.1, 0.2, 0.3],
        filters={"category": "support"},
    )

    assert len(result["documents"]) == 1
    assert result["documents"][0].id == "doc-1"
    assert client.search_calls[0]["filter"] == {"category": "support"}


def test_retriever_converts_haystack_filter():
    client = _FakeClient()
    client.search_calls = []

    def search(**kwargs):
        client.search_calls.append(kwargs)
        return []

    client.search = search

    retriever = DynavecRetriever(client)

    retriever.run(
        query_embedding=[0.1, 0.2, 0.3],
        filters={
            "field": "meta.category",
            "operator": "==",
            "value": "support",
        },
    )

    assert client.search_calls[0]["filter"] == {"category": "support"}

def test_document_store_serialization():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    data = store.to_dict()

    assert data["type"] == "dynavec.integrations.haystack.DynavecDocumentStore"
    assert data["init_parameters"]["namespace"] == "kb"
    assert "config" in data["init_parameters"]

def test_document_store_from_dict(monkeypatch):
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    data = store.to_dict()

    monkeypatch.setattr(
        "dynavec.integrations.haystack.Dynavec",
        lambda config: _FakeClient(),
    )

    restored = DynavecDocumentStore.from_dict(data)

    assert restored.namespace == "kb"

def test_filter_documents_with_haystack_filter():
    client = _FakeClient()
    store = DynavecDocumentStore(client, namespace="kb")

    client.list_vectors = lambda **kwargs: iter([
        type(
            "Result",
            (),
            {
                "id": "doc-1",
                "text": "Password reset",
                "metadata": {"category": "account"},
                "vector": None,
            },
        )(),
        type(
            "Result",
            (),
            {
                "id": "doc-2",
                "text": "Home loan details",
                "metadata": {"category": "loan"},
                "vector": None,
            },
        )(),
    ])

    documents = store.filter_documents(
        filters={
            "field": "meta.category",
            "operator": "==",
            "value": "account",
        }
    )

    assert len(documents) == 1
    assert documents[0].id == "doc-1"
