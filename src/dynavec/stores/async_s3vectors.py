"""Async Amazon S3 Vectors store."""

from __future__ import annotations

import logging
import time
from typing import Any

import aioboto3

from ..config import DynavecConfig
from ..logging import log_store_event
from .s3vectors import _f32, _PUT_LIMIT, _GET_LIMIT, _MAX_TOP_K

Metadata = dict[str, Any]

class AsyncS3VectorsStore:
    _logger = logging.getLogger("dynavec.stores.async_s3vectors")

    def __init__(self, config: DynavecConfig, aioboto_session=None) -> None:
        self._config = config
        self._session = aioboto_session or aioboto3.Session()
        self._client_kwargs: dict[str, object] = {"region_name": config.region}
        botocore_config = config.botocore_config()
        if botocore_config is not None:
            self._client_kwargs["config"] = botocore_config

    async def put_vectors(
        self,
        vectors: list[tuple[str, list[float], Metadata]],
    ) -> None:
        t0 = time.perf_counter()
        async with self._session.client("s3vectors", **self._client_kwargs) as client:
            for start in range(0, len(vectors), _PUT_LIMIT):
                chunk = vectors[start : start + _PUT_LIMIT]
                payload = [
                    {"key": key, "data": {"float32": _f32(vec)}, "metadata": meta}
                    for key, vec, meta in chunk
                ]
                await client.put_vectors(
                    vectorBucketName=self._config.vector_bucket,
                    indexName=self._config.index,
                    vectors=payload,
                )
        log_store_event(
            self._logger,
            "async_s3vectors.put_vectors",
            self._config.structured_logging,
            bucket=self._config.vector_bucket,
            index=self._config.index,
            count=len(vectors),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def _query_kwargs(
        self, query_vector, top_k, filter, return_metadata, return_distance
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

    async def query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        if top_k <= 0:
            return []
        if top_k > _MAX_TOP_K:
            raise ValueError(f"top_k exceeds maximum limit of {_MAX_TOP_K}.")
            
        kwargs = self._query_kwargs(query_vector, top_k, filter, return_metadata, return_distance)
        results: list[dict[str, Any]] = []
        
        async with self._session.client("s3vectors", **self._client_kwargs) as client:
            paginator = client.get_paginator("query_vectors")
            async for page in paginator.paginate(PaginationConfig={"MaxItems": top_k}, **kwargs):
                vectors = page.get("vectors", [])
                results.extend(vectors)
                if len(results) >= top_k:
                    break

        res = results[:top_k]
        log_store_event(
            self._logger,
            "async_s3vectors.query",
            self._config.structured_logging,
            bucket=self._config.vector_bucket,
            index=self._config.index,
            top_k=top_k,
            filtered=filter is not None,
            returned_count=len(res),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return res

    async def get_vectors(
        self, keys: list[str], return_metadata: bool = False
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        async with self._session.client("s3vectors", **self._client_kwargs) as client:
            for start in range(0, len(keys), _GET_LIMIT):
                chunk = keys[start : start + _GET_LIMIT]
                resp = await client.get_vectors(
                    vectorBucketName=self._config.vector_bucket,
                    indexName=self._config.index,
                    keys=chunk,
                    returnData=True,
                    returnMetadata=return_metadata,
                )
                for v in resp.get("vectors", []):
                    out[v["key"]] = {
                        "vector": v.get("data", {}).get("float32"),
                        "metadata": v.get("metadata", {}),
                    }
        return out

    async def delete_vectors(self, keys: list[str]) -> None:
        t0 = time.perf_counter()
        if not keys:
            return
        async with self._session.client("s3vectors", **self._client_kwargs) as client:
            for start in range(0, len(keys), _PUT_LIMIT):
                chunk = keys[start : start + _PUT_LIMIT]
                await client.delete_vectors(
                    vectorBucketName=self._config.vector_bucket,
                    indexName=self._config.index,
                    keys=chunk,
                )
        log_store_event(
            self._logger,
            "async_s3vectors.delete_vectors",
            self._config.structured_logging,
            bucket=self._config.vector_bucket,
            index=self._config.index,
            count=len(keys),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
