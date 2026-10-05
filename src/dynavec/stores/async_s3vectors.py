"""Async Amazon S3 Vectors store backed by aioboto3."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

from ..config import DynavecConfig
from ..logging import log_store_event
from ..utils import TokenBucket, async_retry
from .s3vectors import _GET_LIMIT, _MAX_TOP_K, _PUT_LIMIT, Metadata, _f32


class AsyncS3VectorsStore:
    """Async wrapper around Amazon S3 Vectors I/O."""

    _logger = logging.getLogger("dynavec.stores.async_s3vectors")

    def __init__(self, config: DynavecConfig, boto_session: Any) -> None:
        self._config = config
        self._session = boto_session
        self._put_limiter = TokenBucket(config.put_rps) if config.put_rps is not None else None
        self._query_limiter = (
            TokenBucket(config.query_rps) if config.query_rps is not None else None
        )
        self._client_context: Any | None = None
        self._client: Any | None = None

    async def __aenter__(self) -> AsyncS3VectorsStore:
        if self._client is not None:
            return self

        client_kwargs: dict[str, object] = {"region_name": self._config.region}
        botocore_config = self._config.botocore_config()
        if botocore_config is not None:
            client_kwargs["config"] = botocore_config

        self._client_context = self._session.client(
            "s3vectors",
            **client_kwargs,
        )
        self._client = await self._client_context.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._client_context is not None:
                await self._client_context.__aexit__(
                    exc_type,
                    exc_value,
                    traceback,
                )
        finally:
            self._client = None
            self._client_context = None

    async def aclose(self) -> None:
        try:
            if self._client_context is not None:
                await self._client_context.__aexit__(
                    None,
                    None,
                    None,
                )
        finally:
            self._client = None
            self._client_context = None

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "AsyncS3VectorsStore is not open. "
                "Use it inside 'async with' before performing AWS operations."
            )
        return self._client

    @async_retry()
    async def _put_batch(
        self,
        payload: list[dict[str, Any]],
    ) -> None:
        if self._put_limiter is not None:
            await self._put_limiter.acquire_async()

        client = self._require_client()
        await client.put_vectors(
            vectorBucketName=self._config.vector_bucket,
            indexName=self._config.index,
            vectors=payload,
        )

    async def put_vectors(
        self,
        vectors: list[tuple[str, list[float], Metadata]],
        max_workers: int = 8,
    ) -> None:
        """Insert or overwrite vectors using concurrent async batches."""
        if not vectors:
            return

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0.")

        t0 = time.perf_counter()
        chunks = [vectors[i : i + _PUT_LIMIT] for i in range(0, len(vectors), _PUT_LIMIT)]

        semaphore = asyncio.Semaphore(min(max_workers, len(chunks)))

        async def upload_chunk(
            chunk: list[tuple[str, list[float], Metadata]],
        ) -> None:
            payload = [
                {
                    "key": key,
                    "data": {"float32": _f32(vector)},
                    "metadata": metadata,
                }
                for key, vector, metadata in chunk
            ]

            async with semaphore:
                await self._put_batch(payload)

        tasks = [asyncio.create_task(upload_chunk(chunk)) for chunk in chunks]

        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        log_store_event(
            self._logger,
            "s3vectors.put_vectors",
            self._config.structured_logging,
            bucket=self._config.vector_bucket,
            index=self._config.index,
            count=len(vectors),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def _query_kwargs(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None,
        return_metadata: bool,
        return_distance: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "vectorBucketName": self._config.vector_bucket,
            "indexName": self._config.index,
            "queryVector": {"float32": _f32(query_vector)},
            "topK": top_k,
            "returnMetadata": return_metadata,
            "returnDistance": return_distance,
        }
        if filter:
            kwargs["filter"] = filter
        return kwargs

    @async_retry()
    async def query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
    ) -> list[dict[str, Any]]:
        """Run an ANN query and drain paginated results up to ``top_k``."""
        t0 = time.perf_counter()

        if top_k <= 0:
            log_store_event(
                self._logger,
                "s3vectors.query",
                self._config.structured_logging,
                bucket=self._config.vector_bucket,
                index=self._config.index,
                top_k=top_k,
                filtered=filter is not None,
                returned_count=0,
                duration_ms=0.0,
            )
            return []

        if top_k > _MAX_TOP_K:
            raise ValueError(
                f"top_k ({top_k}) exceeds Amazon S3 Vectors maximum limit of {_MAX_TOP_K}."
            )

        results: list[dict[str, Any]] = []

        async for page in self.query_pages(
            query_vector,
            top_k,
            filter=filter,
            return_metadata=return_metadata,
            return_distance=return_distance,
        ):
            results.extend(page)
            if len(results) >= top_k:
                break

        res = results[:top_k]

        log_store_event(
            self._logger,
            "s3vectors.query",
            self._config.structured_logging,
            bucket=self._config.vector_bucket,
            index=self._config.index,
            top_k=top_k,
            filtered=filter is not None,
            returned_count=len(res),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        return res

    async def query_pages(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
        page_size: int | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield result pages from the async S3 Vectors paginator."""
        if top_k <= 0:
            return

        if top_k > _MAX_TOP_K:
            raise ValueError(
                f"top_k ({top_k}) exceeds Amazon S3 Vectors maximum limit of {_MAX_TOP_K}."
            )

        effective_page_size = page_size if page_size is not None else self._config.top_k_page_size

        if effective_page_size is not None and effective_page_size <= 0:
            raise ValueError("page_size must be a positive integer.")

        kwargs = self._query_kwargs(
            query_vector,
            top_k,
            filter,
            return_metadata,
            return_distance,
        )

        client = self._require_client()
        paginator = client.get_paginator("query_vectors")

        yielded = 0
        buffer: list[dict[str, Any]] = []

        page_iterator = paginator.paginate(
            PaginationConfig={"MaxItems": top_k},
            **kwargs,
        ).__aiter__()

        while True:
            if self._query_limiter is not None:
                await self._query_limiter.acquire_async()

            try:
                page = await page_iterator.__anext__()
            except StopAsyncIteration:
                break

            vectors = page.get("vectors", [])

            if vectors:
                if effective_page_size is None:
                    remaining = top_k - yielded

                    if len(vectors) > remaining:
                        vectors = vectors[:remaining]

                    yield vectors
                    yielded += len(vectors)

                    if yielded >= top_k:
                        return

                else:
                    buffer.extend(vectors)

                    while len(buffer) >= effective_page_size and yielded < top_k:
                        chunk = buffer[:effective_page_size]
                        buffer = buffer[effective_page_size:]

                        remaining = top_k - yielded

                        if len(chunk) > remaining:
                            chunk = chunk[:remaining]

                        yield chunk
                        yielded += len(chunk)

                        if yielded >= top_k:
                            return

            if page.get("NextToken") is None:
                break

        if effective_page_size is not None and buffer and yielded < top_k:
            remaining = top_k - yielded
            yield buffer[:remaining]

    @async_retry()
    async def get_vectors(
        self,
        keys: list[str],
        return_metadata: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Fetch stored vectors by key for MMR reranking."""
        client = self._require_client()

        out: dict[str, dict[str, Any]] = {}

        for start in range(0, len(keys), _GET_LIMIT):
            chunk = keys[start : start + _GET_LIMIT]

            response = await client.get_vectors(
                vectorBucketName=self._config.vector_bucket,
                indexName=self._config.index,
                keys=chunk,
                returnData=True,
                returnMetadata=return_metadata,
            )

            for vector in response.get("vectors", []):
                out[vector["key"]] = {
                    "vector": vector.get("data", {}).get("float32"),
                    "metadata": vector.get("metadata", {}),
                }

        return out
