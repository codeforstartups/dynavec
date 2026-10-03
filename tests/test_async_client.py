"""Tests for the high-level AsyncDynavec client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from dynavec.async_client import AsyncDynavec
from dynavec.client_common import s3_key
from dynavec.config import NS_METADATA_KEY, DynavecConfig
from dynavec.embeddings.base import Embedder
from dynavec.exceptions import (
    ConfigurationError,
    DimensionMismatchError,
)


def _config() -> DynavecConfig:
    return DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="docs",
        dimension=4,
    )


class AsyncOnlyEmbedder(Embedder):
    dimension = 4

    def __init__(self) -> None:
        self.async_calls: list[list[str]] = []
        self.async_query_calls: list[str] = []

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        raise AssertionError(
            "AsyncDynavec must not call sync embed_query()."
        )

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:
        self.async_query_calls.append(text)
        return [1.0, 2.0, 3.0, 4.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise AssertionError(
            "AsyncDynavec must not call sync embed_documents()."
        )

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.async_calls.append(texts)

        return [
            [1.0, 2.0, 3.0, 4.0]
            for _ in texts
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.put_calls: list[
            tuple[list[tuple[str, list[float], dict[str, Any]]], int]
        ] = []
        self.query_results: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.query_page_results: list[list[dict[str, Any]]] = []
        self.query_page_calls: list[dict[str, Any]] = []

    async def query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
    ) -> list[dict[str, Any]]:
        self.query_calls.append(
            {
                "query_vector": query_vector,
                "top_k": top_k,
                "filter": filter,
                "return_metadata": return_metadata,
                "return_distance": return_distance,
            }
        )

        return self.query_results

    async def __aenter__(self) -> FakeVectorStore:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def put_vectors(
        self,
        vectors: list[
            tuple[str, list[float], dict[str, Any]]
        ],
        max_workers: int = 8,
    ) -> None:
        self.put_calls.append(
            (vectors, max_workers)
        )

    async def query_pages(
        self,
        query_vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
        page_size: int | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        self.query_page_calls.append(
            {
                "query_vector": query_vector,
                "top_k": top_k,
                "filter": filter,
                "return_metadata": return_metadata,
                "return_distance": return_distance,
                "page_size": page_size,
            }
        )

        for page in self.query_page_results:
            yield page


class FakeDocumentStore:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.put_calls: list[
            tuple[
                str,
                list[
                    tuple[
                        str,
                        str | None,
                        dict[str, Any],
                    ]
                ],
            ]
        ] = []
        self.documents: dict[str, dict[str, Any]] = {}
        self.get_calls: list[tuple[str, list[str]]] = []

    async def get_many(
        self,
        namespace: str,
        ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        self.get_calls.append(
            (
                namespace,
                ids,
            )
        )

        return {
            doc_id: self.documents[doc_id]
            for doc_id in ids
            if doc_id in self.documents
        }

    async def __aenter__(
        self,
    ) -> FakeDocumentStore:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def put_many(
        self,
        namespace: str,
        items: list[
            tuple[
                str,
                str | None,
                dict[str, Any],
            ]
        ],
    ) -> None:
        self.put_calls.append(
            (namespace, items)
        )


def _install_fake_stores(
    client: AsyncDynavec,
    vectors: FakeVectorStore,
    documents: FakeDocumentStore,
) -> None:
    client._vectors = vectors  # type: ignore[assignment]
    client._docs = documents  # type: ignore[assignment]


def _client(
    *,
    embedder: Embedder | None = None,
) -> tuple[
    AsyncDynavec,
    FakeVectorStore,
    FakeDocumentStore,
]:
    client = AsyncDynavec(
        _config(),
        embedder=embedder,
        boto_session=object(),
    )

    vectors = FakeVectorStore()
    documents = FakeDocumentStore()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    return client, vectors, documents


async def test_async_context_opens_and_closes_stores() -> None:
    client, vectors, documents = _client()

    async with client:
        assert vectors.entered
        assert documents.entered
        assert not vectors.exited
        assert not documents.exited

    assert vectors.exited
    assert documents.exited


async def test_aclose_closes_open_stores() -> None:
    client, vectors, documents = _client()

    await client.__aenter__()

    assert vectors.entered
    assert documents.entered

    await client.aclose()

    assert vectors.exited
    assert documents.exited

    # Closing twice should be safe.
    await client.aclose()


async def test_aupsert_requires_open_client() -> None:
    client, _, _ = _client()

    with pytest.raises(
        RuntimeError,
        match="not open",
    ):
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ]
        )


async def test_aupsert_uses_async_embedding_and_writes_both_stores() -> None:
    embedder = AsyncOnlyEmbedder()
    client, vectors, documents = _client(
        embedder=embedder
    )

    async with client:
        result = await client.aupsert(
            [
                {
                    "id": "a",
                    "text": "hello",
                    "metadata": {
                        "topic": "test",
                    },
                }
            ],
            namespace="tenant",
        )

    assert result.count == 1
    assert result.ids == ["a"]

    assert embedder.async_calls == [
        ["hello"]
    ]

    assert len(vectors.put_calls) == 1
    assert len(documents.put_calls) == 1

    vector_payload, _ = vectors.put_calls[0]

    assert len(vector_payload) == 1
    assert vector_payload[0][1] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    namespace, document_payload = (
        documents.put_calls[0]
    )

    assert namespace == "tenant"
    assert document_payload[0][0] == "a"
    assert document_payload[0][1] == "hello"


async def test_aupsert_rejects_mismatched_embedding_count_before_writing() -> None:
    class ShortEmbedder(AsyncOnlyEmbedder):
        async def aembed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [[1.0, 2.0, 3.0, 4.0]]

    client, vectors, documents = _client(embedder=ShortEmbedder())

    async with client:
        with pytest.raises(
            ConfigurationError,
            match=r"Embedder returned 1 vectors for 2 documents",
        ):
            await client.aupsert(
                [
                    {"id": "a", "text": "hello"},
                    {"id": "b", "text": "world"},
                ]
            )

    assert vectors.put_calls == []
    assert documents.put_calls == []


async def test_aupsert_passes_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, documents = _client()

    monkeypatch.setattr(
        "dynavec.client_common.time.time",
        lambda: 1_700_000_000,
    )

    async with client:
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            namespace="tenant",
            ttl_seconds=60,
        )

    namespace, document_payload = documents.put_calls[0]

    assert namespace == "tenant"
    assert document_payload[0][2]["_ttl"] == 1_700_000_060


async def test_aupsert_rejects_non_positive_ttl() -> None:
    client, _, _ = _client()

    with pytest.raises(
        ValueError,
        match="ttl_seconds must be positive",
    ):
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            ttl_seconds=0,
        )


async def test_aupsert_runs_s3_and_dynamodb_writes_concurrently() -> None:
    s3_started = asyncio.Event()
    ddb_started = asyncio.Event()

    class CoordinatedVectors(
        FakeVectorStore
    ):
        async def put_vectors(
            self,
            vectors: list[
                tuple[
                    str,
                    list[float],
                    dict[str, Any],
                ]
            ],
            max_workers: int = 8,
        ) -> None:
            s3_started.set()

            await asyncio.wait_for(
                ddb_started.wait(),
                timeout=1,
            )

    class CoordinatedDocuments(
        FakeDocumentStore
    ):
        async def put_many(
            self,
            namespace: str,
            items: list[
                tuple[
                    str,
                    str | None,
                    dict[str, Any],
                ]
            ],
        ) -> None:
            ddb_started.set()

            await asyncio.wait_for(
                s3_started.wait(),
                timeout=1,
            )

    client = AsyncDynavec(
        _config(),
        boto_session=object(),
    )

    vectors = CoordinatedVectors()
    documents = CoordinatedDocuments()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    async with client:
        await asyncio.wait_for(
            client.aupsert(
                [
                    {
                        "id": "a",
                        "vector": [
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                        ],
                    }
                ]
            ),
            timeout=1,
        )

    assert s3_started.is_set()
    assert ddb_started.is_set()


async def test_partial_enter_failure_closes_first_store() -> None:
    vectors = FakeVectorStore()

    class FailingDocumentStore(
        FakeDocumentStore
    ):
        async def __aenter__(
            self,
        ) -> FailingDocumentStore:
            raise RuntimeError(
                "DynamoDB open failed"
            )

    client = AsyncDynavec(
        _config(),
        boto_session=object(),
    )

    documents = FailingDocumentStore()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    with pytest.raises(
        RuntimeError,
        match="DynamoDB open failed",
    ):
        async with client:
            pass

    assert vectors.entered
    assert vectors.exited


async def test_resolve_query_vector_uses_precomputed_vector():
    client, _, _ = _client()

    vector = [0.1, 0.2, 0.3, 0.4]

    result = await client._resolve_query_vector(
        None,
        vector,
    )

    assert result is vector


async def test_resolve_query_vector_rejects_wrong_dimension():
    client, _, _ = _client()

    with pytest.raises(
        DimensionMismatchError,
        match="Query vector dimension",
    ):
        await client._resolve_query_vector(
            None,
            [0.1, 0.2],
        )


async def test_resolve_query_vector_requires_query_or_vector():
    client, _, _ = _client()

    with pytest.raises(
        ValueError,
        match="Provide either 'query' text or a 'vector'",
    ):
        await client._resolve_query_vector(
            None,
            None,
        )


async def test_resolve_query_vector_requires_embedder():
    client, _, _ = _client()

    with pytest.raises(
        ConfigurationError,
        match="Text query requires an embedder",
    ):
        await client._resolve_query_vector(
            "hello",
            None,
        )


async def test_resolve_query_vector_uses_async_embedder():
    embedder = AsyncOnlyEmbedder()
    client, _, _ = _client(embedder=embedder)

    result = await client._resolve_query_vector(
        "hello",
        None,
    )

    assert result == [1.0, 2.0, 3.0, 4.0]
    assert embedder.async_query_calls == ["hello"]


async def test_asearch_queries_and_hydrates_results():
    client, vectors, documents = _client()

    vectors.query_results = [
        {
            "key": s3_key("default", "doc-1"),
            "distance": 0.2,
        },
        {
            "key": s3_key("default", "doc-2"),
            "distance": 0.4,
        },
    ]

    documents.documents = {
        "doc-1": {
            "text": "First document",
            "metadata": {"topic": "python"},
            "ttl": 1_700_000_060,
        },
        "doc-2": {
            "text": "Second document",
            "metadata": {"topic": "asyncio"},
        },
    }

    async with client:
        results = await client.asearch(
            vector=[0.1, 0.2, 0.3, 0.4],
            top_k=2,
        )

    assert [result.id for result in results] == [
        "doc-1",
        "doc-2",
    ]

    assert results[0].distance == 0.2
    assert results[0].text == "First document"
    assert results[0].metadata == {"topic": "python"}
    assert results[0].ttl == 1_700_000_060

    assert results[1].distance == 0.4
    assert results[1].text == "Second document"
    assert results[1].metadata == {"topic": "asyncio"}
    assert results[1].ttl is None

    assert documents.get_calls == [
        (
            "default",
            ["doc-1", "doc-2"],
        )
    ]

    assert len(vectors.query_calls) == 1
    assert vectors.query_calls[0]["query_vector"] == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    assert vectors.query_calls[0]["top_k"] == 2


async def test_asearch_returns_empty_when_vector_search_has_no_hits():
    client, vectors, documents = _client()

    vectors.query_results = []

    async with client:
        results = await client.asearch(
            vector=[0.1, 0.2, 0.3, 0.4],
        )

    assert results == []
    assert documents.get_calls == []


async def test_asearch_passes_namespace_and_filter_to_vector_store():
    client, vectors, _ = _client()

    vectors.query_results = []

    async with client:
        await client.asearch(
            vector=[0.1, 0.2, 0.3, 0.4],
            namespace="tenant-a",
            filter={"topic": "python"},
        )

    assert vectors.query_calls[0]["filter"] == {
        "$and": [
            {"topic": "python"},
            {NS_METADATA_KEY: "tenant-a"},
        ]
    }


async def test_asearch_stream_yields_results_page_by_page():
    client, vectors, documents = _client()

    vectors.query_page_results = [
        [
            {
                "key": s3_key("tenant-a", "doc-1"),
                "distance": 0.1,
            },
            {
                "key": s3_key("tenant-a", "doc-2"),
                "distance": 0.2,
            },
        ],
        [
            {
                "key": s3_key("tenant-a", "doc-3"),
                "distance": 0.3,
            }
        ],
    ]

    documents.documents = {
        "doc-1": {
            "text": "First",
            "metadata": {"topic": "python"},
        },
        "doc-2": {
            "text": "Second",
            "metadata": {"topic": "python"},
        },
        "doc-3": {
            "text": "Third",
            "metadata": {"topic": "python"},
        },
    }

    async with client:
        results = [
            result
            async for result in client.asearch_stream(
                vector=[0.1, 0.2, 0.3, 0.4],
                top_k=3,
                namespace="tenant-a",
                filter={"topic": "python"},
                page_size=2,
            )
        ]

    assert [result.id for result in results] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]

    assert documents.get_calls == [
        (
            "tenant-a",
            ["doc-1", "doc-2"],
        ),
        (
            "tenant-a",
            ["doc-3"],
        ),
    ]

    assert len(vectors.query_page_calls) == 1

    call = vectors.query_page_calls[0]

    assert call["top_k"] == 3
    assert call["page_size"] == 2
    assert call["filter"] == {
        "$and": [
            {"topic": "python"},
            {NS_METADATA_KEY: "tenant-a"},
        ]
    }


async def test_asearch_stream_stops_at_top_k():
    client, vectors, documents = _client()

    vectors.query_page_results = [
        [
            {
                "key": s3_key("default", "doc-1"),
                "distance": 0.1,
            },
            {
                "key": s3_key("default", "doc-2"),
                "distance": 0.2,
            },
            {
                "key": s3_key("default", "doc-3"),
                "distance": 0.3,
            },
        ]
    ]

    documents.documents = {
        "doc-1": {"text": "First", "metadata": {}},
        "doc-2": {"text": "Second", "metadata": {}},
        "doc-3": {"text": "Third", "metadata": {}},
    }

    async with client:
        results = [
            result
            async for result in client.asearch_stream(
                vector=[0.1, 0.2, 0.3, 0.4],
                top_k=2,
            )
        ]

    assert [result.id for result in results] == [
        "doc-1",
        "doc-2",
    ]
