"""Tests for async embedding support in dynavec.embeddings."""

from __future__ import annotations

import asyncio
import threading
from typing import ClassVar

import pytest

from dynavec.embeddings.base import Embedder, Vector


class _SyncTrackingEmbedder(Embedder):
    """Test embedder that tracks call counts and thread IDs."""

    dimension: ClassVar[int] = 3

    def __init__(self) -> None:
        self.embed_docs_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self.thread_ids: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        self.thread_ids.append(threading.get_ident())
        self.embed_docs_calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> Vector:
        self.thread_ids.append(threading.get_ident())
        self.embed_query_calls.append(text)
        return [float(len(text)), 2.0, 0.0]


class _NativeAsyncEmbedder(Embedder):
    """Test embedder providing native async implementation."""

    dimension: ClassVar[int] = 2

    def __init__(self) -> None:
        self.native_async_docs_called = False
        self.native_async_query_called = False

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        raise NotImplementedError("Sync embed_documents should not be called")

    async def aembed_documents(self, texts: list[str]) -> list[Vector]:
        self.native_async_docs_called = True
        await asyncio.sleep(0.001)
        return [[float(len(t)), 99.0] for t in texts]

    async def aembed_query(self, text: str) -> Vector:
        self.native_async_query_called = True
        await asyncio.sleep(0.001)
        return [float(len(text)), 100.0]


class _FailingEmbedder(Embedder):
    """Test embedder that raises an error during embedding."""

    dimension: ClassVar[int] = 2

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        raise RuntimeError("Embedding service unavailable")


@pytest.mark.asyncio
async def test_aembed_documents_default_thread_offloading():
    embedder = _SyncTrackingEmbedder()
    current_thread_id = threading.get_ident()

    vectors = await embedder.aembed_documents(["apple", "banana", "cherry"])

    assert len(vectors) == 3
    assert vectors[0] == [5.0, 1.0, 0.0]
    assert vectors[1] == [6.0, 1.0, 0.0]
    assert vectors[2] == [6.0, 1.0, 0.0]
    assert embedder.embed_docs_calls == [["apple", "banana", "cherry"]]

    # Execution happened on a worker thread, offloading the event loop
    assert len(embedder.thread_ids) == 1
    assert embedder.thread_ids[0] != current_thread_id


@pytest.mark.asyncio
async def test_aembed_query_default_thread_offloading():
    embedder = _SyncTrackingEmbedder()
    current_thread_id = threading.get_ident()

    vector = await embedder.aembed_query("search query")

    assert vector == [12.0, 2.0, 0.0]
    assert embedder.embed_query_calls == ["search query"]
    assert len(embedder.thread_ids) == 1
    assert embedder.thread_ids[0] != current_thread_id


@pytest.mark.asyncio
async def test_aembed_documents_empty():
    embedder = _SyncTrackingEmbedder()
    vectors = await embedder.aembed_documents([])
    assert vectors == []
    assert embedder.embed_docs_calls == [[]]


@pytest.mark.asyncio
async def test_aembed_concurrent_gathering():
    embedder = _SyncTrackingEmbedder()
    queries = [f"query_{i}" for i in range(10)]

    results = await asyncio.gather(*(embedder.aembed_query(q) for q in queries))

    assert len(results) == 10
    for i, res in enumerate(results):
        expected_len = float(len(f"query_{i}"))
        assert res == [expected_len, 2.0, 0.0]


@pytest.mark.asyncio
async def test_native_async_override():
    embedder = _NativeAsyncEmbedder()

    docs = await embedder.aembed_documents(["foo", "barbaz"])
    assert embedder.native_async_docs_called is True
    assert docs == [[3.0, 99.0], [6.0, 99.0]]

    query_vec = await embedder.aembed_query("test")
    assert embedder.native_async_query_called is True
    assert query_vec == [4.0, 100.0]


@pytest.mark.asyncio
async def test_aembed_exception_propagation():
    embedder = _FailingEmbedder()

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        await embedder.aembed_documents(["doc"])

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        await embedder.aembed_query("query")


@pytest.mark.asyncio
async def test_aembed_cancellation():
    class _SlowEmbedder(Embedder):
        dimension: ClassVar[int] = 1

        def embed_documents(self, texts: list[str]) -> list[Vector]:
            import time

            time.sleep(0.5)
            return [[1.0] for _ in texts]

    embedder = _SlowEmbedder()
    task = asyncio.create_task(embedder.aembed_documents(["slow"]))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
