"""Tests for the Semantic Kernel vector store integration."""

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel.data.vector import (  # noqa: E402
    FieldTypes,
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

from dynavec.integrations.semantic_kernel import (  # noqa: E402
    DynavecCollection,
    DynavecStore,
)
from dynavec.models import SearchResult  # noqa: E402


class _FakeResult:
    ids = ["doc-1"]


class _FakeClient:
    def __init__(self):
        self.calls = []

    def upsert(self, documents, **kwargs):
        self.calls.append(("upsert", documents, kwargs))
        return _FakeResult()

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [
            SearchResult(
                id="doc-1",
                score=0.95,
                text="Banking knowledge",
                metadata={"category": "banking"},
                vector=[0.1, 0.2, 0.3],
            ),
            SearchResult(
                id="doc-2",
                score=0.85,
                text="Finance knowledge",
                metadata={"category": "finance"},
                vector=[0.4, 0.5, 0.6],
            ),
        ]

    def get(self, ids, **kwargs):
        self.calls.append(("get", ids, kwargs))
        return [
            SearchResult(
                id="doc-1",
                score=1.0,
                text="Banking knowledge",
                metadata={"category": "banking"},
                vector=[0.1, 0.2, 0.3],
            )
        ]

    def delete(self, ids, **kwargs):
        self.calls.append(("delete", ids, kwargs))


def _definition():
    return VectorStoreCollectionDefinition(
        fields=[
            VectorStoreField(
                field_type=FieldTypes.KEY,
                name="id",
            ),
            VectorStoreField(
                field_type=FieldTypes.DATA,
                name="text",
            ),
            VectorStoreField(
                field_type=FieldTypes.DATA,
                name="category",
            ),
            VectorStoreField(
                field_type=FieldTypes.VECTOR,
                name="embedding",
                dimensions=3,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_upsert():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    ids = await collection.upsert(
        [
            {
                "id": "doc-1",
                "text": "Banking knowledge",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    )

    assert ids == ["doc-1"]

    _, documents, kwargs = client.calls[0]
    assert len(documents) == 1
    assert documents[0].id == "doc-1"
    assert documents[0].text == "Banking knowledge"
    assert documents[0].vector == [0.1, 0.2, 0.3]
    assert kwargs == {"namespace": "kb"}


@pytest.mark.asyncio
async def test_search():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    results = await collection.search(
        values="banking",
        top=2,
    )

    search_results = [result async for result in results.results]

    assert len(search_results) == 2

    first = search_results[0]
    assert first.record["id"] == "doc-1"
    assert first.record["text"] == "Banking knowledge"
    assert first.record["category"] == "banking"
    assert first.score == 0.95


@pytest.mark.asyncio
async def test_search_with_filter():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    results = await collection.search(
        values="banking",
        filter=lambda x: x["category"] == "banking",
        top=2,
    )

    search_results = [result async for result in results.results]

    assert len(search_results) == 2
    assert client.calls[-1] == (
        "search",
        {
            "query": "banking",
            "top_k": 2,
            "namespace": "kb",
            "filter": {"category": "banking"},
            "include_vectors": False,
        },
    )


@pytest.mark.asyncio
async def test_get():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    results = await collection.get(["doc-1"])

    assert len(results) == 1
    assert results[0]["id"] == "doc-1"
    assert results[0]["text"] == "Banking knowledge"
    assert results[0]["category"] == "banking"
    assert client.calls[-1] == (
        "get",
        ["doc-1"],
        {"namespace": "kb"},
    )


@pytest.mark.asyncio
async def test_delete():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    await collection.delete(["doc-1"])

    assert client.calls[-1] == (
        "delete",
        ["doc-1"],
        {"namespace": "kb"},
    )


@pytest.mark.asyncio
async def test_search_with_gte_filter():
    client = _FakeClient()
    collection = DynavecCollection(
        client=client,
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    results = await collection.search(
        values="banking",
        filter=lambda x: x["score"] >= 80,
        top=2,
    )

    [result async for result in results.results]

    assert client.calls[-1] == (
        "search",
        {
            "query": "banking",
            "top_k": 2,
            "namespace": "kb",
            "filter": {"score": {"$gte": 80}},
            "include_vectors": False,
        },
    )


def test_store_get_collection():
    client = _FakeClient()
    store = DynavecStore(client=client)

    collection = store.get_collection(
        record_type=dict,
        definition=_definition(),
        collection_name="kb",
    )

    assert isinstance(collection, DynavecCollection)
    assert collection._client is client
    assert collection._namespace == "kb"


@pytest.mark.asyncio
async def test_store_list_collection_names():
    store = DynavecStore(client=_FakeClient())

    names = await store.list_collection_names()

    assert names == []
