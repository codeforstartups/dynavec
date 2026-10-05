"""Tests for the async S3 Vectors store."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dynavec.config import DynavecConfig
from dynavec.stores.async_s3vectors import AsyncS3VectorsStore


class FakeAsyncPaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    async def paginate(self, **kwargs: Any):
        self.calls.append(kwargs)

        for page in self.pages:
            yield page


def _config() -> DynavecConfig:
    return DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
    )


class FakeClientContext:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Any:
        self.entered = True
        return self.client

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True


class FakeSession:
    def __init__(self, client: Any) -> None:
        self.client_instance = client
        self.context: FakeClientContext | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def client(self, service_name: str, **kwargs: Any) -> FakeClientContext:
        self.calls.append((service_name, kwargs))
        self.context = FakeClientContext(self.client_instance)
        return self.context


class FakeS3Client:
    def __init__(
        self,
        pages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.paginator = FakeAsyncPaginator(pages or [])

    async def put_vectors(self, **kwargs: Any) -> None:
        self.put_calls.append(kwargs)

    def get_paginator(self, name: str) -> FakeAsyncPaginator:
        assert name == "query_vectors"
        return self.paginator

    async def get_vectors(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)

        return {
            "vectors": [
                {
                    "key": key,
                    "data": {"float32": [1.0, 2.0, 3.0, 4.0]},
                    "metadata": {"source": "test"},
                }
                for key in kwargs["keys"]
            ]
        }


async def test_async_store_enters_and_closes_client():
    client = FakeS3Client()
    session = FakeSession(client)

    async with AsyncS3VectorsStore(_config(), session):
        assert session.context is not None
        assert session.context.entered

    assert session.context is not None
    assert session.context.exited
    assert session.calls[0][0] == "s3vectors"


async def test_put_vectors_batches_at_service_limit():
    client = FakeS3Client()
    session = FakeSession(client)

    vectors = [(f"key-{i}", [0.1, 0.2, 0.3, 0.4], {"tag": "test"}) for i in range(1200)]

    async with AsyncS3VectorsStore(_config(), session) as store:
        await store.put_vectors(vectors)

    assert sorted(len(call["vectors"]) for call in client.put_calls) == [
        200,
        500,
        500,
    ]


async def test_put_vectors_runs_batches_concurrently(monkeypatch):
    session = FakeSession(FakeS3Client())

    async with AsyncS3VectorsStore(_config(), session) as store:
        active = 0
        max_active = 0

        async def fake_put_batch(payload):
            nonlocal active, max_active

            active += 1
            max_active = max(max_active, active)

            await asyncio.sleep(0)

            active -= 1

        monkeypatch.setattr(store, "_put_batch", fake_put_batch)

        vectors = [(f"key-{i}", [0.1, 0.2, 0.3, 0.4], {}) for i in range(1200)]

        await store.put_vectors(vectors)

    assert max_active > 1


async def test_put_vectors_propagates_errors(monkeypatch):
    session = FakeSession(FakeS3Client())

    async with AsyncS3VectorsStore(_config(), session) as store:

        async def fail(payload):
            raise RuntimeError("S3 API failure")

        monkeypatch.setattr(store, "_put_batch", fail)

        with pytest.raises(RuntimeError, match="S3 API failure"):
            await store.put_vectors([("key", [0.1, 0.2, 0.3, 0.4], {})])


async def test_put_vectors_requires_open_store():
    store = AsyncS3VectorsStore(_config(), FakeSession(FakeS3Client()))

    with pytest.raises(RuntimeError, match="not open"):
        await store.put_vectors([("key", [0.1, 0.2, 0.3, 0.4], {})])


async def test_query_returns_paginated_results():
    pages = [
        {
            "vectors": [{"key": f"k{i}", "distance": i * 0.01} for i in range(100)],
            "NextToken": "token-1",
        },
        {"vectors": [{"key": f"k{i}", "distance": i * 0.01} for i in range(100, 150)]},
    ]

    client = FakeS3Client(pages)
    session = FakeSession(client)

    async with AsyncS3VectorsStore(_config(), session) as store:
        results = await store.query(
            [0.1, 0.2, 0.3, 0.4],
            top_k=150,
        )

    assert len(results) == 150
    assert results[0]["key"] == "k0"
    assert results[-1]["key"] == "k149"


async def test_query_truncates_at_top_k():
    pages = [
        {
            "vectors": [{"key": f"k{i}"} for i in range(100)],
            "NextToken": "token-1",
        },
        {"vectors": [{"key": f"k{i}"} for i in range(100, 200)]},
    ]

    client = FakeS3Client(pages)
    session = FakeSession(client)

    async with AsyncS3VectorsStore(_config(), session) as store:
        results = await store.query(
            [0.1, 0.2, 0.3, 0.4],
            top_k=130,
        )

    assert len(results) == 130
    assert results[-1]["key"] == "k129"


async def test_query_pages_respects_page_size():
    pages = [{"vectors": [{"key": f"k{i}"} for i in range(25)]}]

    client = FakeS3Client(pages)
    session = FakeSession(client)

    async with AsyncS3VectorsStore(_config(), session) as store:
        result_pages = [
            page
            async for page in store.query_pages(
                [0.1, 0.2, 0.3, 0.4],
                top_k=25,
                page_size=10,
            )
        ]

    assert [len(page) for page in result_pages] == [10, 10, 5]


async def test_query_rejects_excessive_top_k():
    client = FakeS3Client()
    session = FakeSession(client)

    async with AsyncS3VectorsStore(_config(), session) as store:
        with pytest.raises(
            ValueError,
            match="exceeds Amazon S3 Vectors maximum limit",
        ):
            await store.query(
                [0.1, 0.2, 0.3, 0.4],
                top_k=10_001,
            )


async def test_put_batch_uses_rate_limiter():
    config = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        put_rps=10,
    )

    client = FakeS3Client()
    session = FakeSession(client)

    async with AsyncS3VectorsStore(config, session) as store:
        limiter = AsyncMock()
        store._put_limiter = limiter

        payload = [
            {
                "key": "key-1",
                "data": {
                    "float32": [0.1, 0.2, 0.3, 0.4],
                },
                "metadata": {},
            }
        ]

        await store._put_batch(payload)

    limiter.acquire_async.assert_awaited_once_with()
    assert len(client.put_calls) == 1


async def test_query_pages_uses_rate_limiter_per_page():
    pages = [
        {
            "vectors": [{"key": "k0"}],
            "NextToken": "token-1",
        },
        {
            "vectors": [{"key": "k1"}],
            "NextToken": "token-2",
        },
        {
            "vectors": [{"key": "k2"}],
        },
    ]

    config = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        query_rps=10,
    )

    client = FakeS3Client(pages)
    session = FakeSession(client)

    async with AsyncS3VectorsStore(config, session) as store:
        limiter = AsyncMock()
        store._query_limiter = limiter

        result_pages = [
            page
            async for page in store.query_pages(
                [0.1, 0.2, 0.3, 0.4],
                top_k=3,
            )
        ]

    assert [len(page) for page in result_pages] == [1, 1, 1]
    assert limiter.acquire_async.await_count == 3


async def test_get_vectors_batches_keys():
    client = FakeS3Client()
    session = FakeSession(client)

    keys = [f"key-{i}" for i in range(250)]

    async with AsyncS3VectorsStore(_config(), session) as store:
        result = await store.get_vectors(
            keys,
            return_metadata=True,
        )

    assert [len(call["keys"]) for call in client.get_calls] == [
        100,
        100,
        50,
    ]

    assert all(call["returnData"] is True for call in client.get_calls)
    assert all(call["returnMetadata"] is True for call in client.get_calls)

    assert len(result) == 250
    assert set(result) == set(keys)
