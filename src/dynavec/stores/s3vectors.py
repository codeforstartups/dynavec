"""Amazon S3 Vectors store: the serverless, billion-scale ANN tier.

Wraps the boto3 ``s3vectors`` client. S3 Vectors owns the approximate-nearest-
neighbor index (AWS-managed, ~90%+ recall, ~100ms warm / sub-second cold). We
store the vector + a *small filterable* metadata subset here; the full document
lives in DynamoDB.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..config import DynavecConfig
from ..utils import retry

Metadata = dict[str, Any]

# PutVectors accepts up to 500 vectors per call.
_PUT_LIMIT = 500
_GET_LIMIT = 100
# QueryVectors topK maximum (results are returned in pages of at most 100).
_MAX_TOP_K = 10_000
# ListVectors maxResults range.
_LIST_PAGE_MIN = 1
_LIST_PAGE_MAX = 1000


def _f32(vector: list[float]) -> list[float]:
    """Ensure vector is float32-clean (S3 Vectors stores float32)."""
    return np.asarray(vector, dtype=np.float32).tolist()


class S3VectorsStore:
    def __init__(self, config: DynavecConfig, boto_session=None) -> None:
        import boto3  # local import: base import stays cheap

        session = boto_session or boto3.Session()
        self._config = config
        self._client = session.client("s3vectors", region_name=config.region)

    @retry()
    def _put_batch(self, payload: list[dict]) -> None:
        self._client.put_vectors(
            vectorBucketName=self._config.vector_bucket,
            indexName=self._config.index,
            vectors=payload,
        )

    def put_vectors(
        self,
        vectors: list[tuple[str, list[float], Metadata]],
    ) -> None:
        """Insert/overwrite (key, vector, filterable_metadata) triples."""
        for start in range(0, len(vectors), _PUT_LIMIT):
            chunk = vectors[start : start + _PUT_LIMIT]
            payload = [
                {"key": key, "data": {"float32": _f32(vec)}, "metadata": meta}
                for key, vec, meta in chunk
            ]
            self._put_batch(payload)

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

    @retry()
    def query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
    ) -> list[dict[str, Any]]:
        """Run an ANN query. Drains paginated results up to ``top_k``."""
        if top_k <= 0:
            return []
        if top_k > _MAX_TOP_K:
            raise ValueError(
                f"top_k ({top_k}) exceeds Amazon S3 Vectors maximum limit of {_MAX_TOP_K}."
            )
        results: list[dict[str, Any]] = []
        for page in self.query_pages(
            query_vector,
            top_k,
            filter=filter,
            return_metadata=return_metadata,
            return_distance=return_distance,
        ):
            results.extend(page)
            if len(results) >= top_k:
                break
        return results[:top_k]

    def query_pages(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Metadata | None = None,
        return_metadata: bool = True,
        return_distance: bool = True,
        page_size: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield result **pages** as the S3 Vectors paginator returns them.

        This is what powers streaming search: an agent can begin consuming the
        first page while later pages are still in flight.

        Parameters
        ----------
        page_size:
            Optional client-side chunk size for yielded pages. When None (default),
            yields the raw pages returned by Amazon S3 Vectors (up to 100 per page).
            This does not change the service page size.
        """
        if top_k <= 0:
            return
        if top_k > _MAX_TOP_K:
            raise ValueError(
                f"top_k ({top_k}) exceeds Amazon S3 Vectors maximum limit of {_MAX_TOP_K}."
            )

        effective_page_size = (
            page_size if page_size is not None else self._config.top_k_page_size
        )
        if effective_page_size is not None and effective_page_size <= 0:
            raise ValueError("page_size must be a positive integer.")

        kwargs = self._query_kwargs(
            query_vector, top_k, filter, return_metadata, return_distance
        )
        paginator = self._client.get_paginator("query_vectors")
        yielded = 0
        buffer: list[dict[str, Any]] = []

        for page in paginator.paginate(
            PaginationConfig={"MaxItems": top_k},
            **kwargs,
        ):
            vectors = page.get("vectors", [])
            if not vectors:
                continue

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

        if effective_page_size is not None and buffer and yielded < top_k:
            remaining = top_k - yielded
            yield buffer[:remaining]

    def list_pages(
        self,
        return_data: bool = False,
        return_metadata: bool = False,
        page_size: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield ``ListVectors`` **pages** as the boto3 paginator returns them.

        A full scan of the index, not a query: there is no query vector and no
        server-side filter (Amazon S3 Vectors ``ListVectors`` accepts no
        ``filter`` parameter), so callers scope the results themselves. The
        paginator drives the ``nextToken`` continuation and stops when the
        service stops returning one.

        Parameters
        ----------
        page_size:
            Service-side page size (``maxResults``, 1-1000). None (default)
            leaves the page size to Amazon S3 Vectors.
        """
        if page_size is not None and not _LIST_PAGE_MIN <= page_size <= _LIST_PAGE_MAX:
            raise ValueError(
                f"page_size must be between {_LIST_PAGE_MIN} and {_LIST_PAGE_MAX}."
            )

        kwargs: dict[str, Any] = {
            "vectorBucketName": self._config.vector_bucket,
            "indexName": self._config.index,
            "returnData": return_data,
            "returnMetadata": return_metadata,
        }
        pagination_config: dict[str, Any] = {}
        if page_size is not None:
            pagination_config["PageSize"] = page_size

        paginator = self._client.get_paginator("list_vectors")
        for page in paginator.paginate(PaginationConfig=pagination_config, **kwargs):
            vectors = page.get("vectors", [])
            if vectors:
                yield vectors

    @retry()
    def get_vectors(
        self, keys: list[str], return_metadata: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Fetch stored vectors by key (used to feed MMR reranking)."""
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(keys), _GET_LIMIT):
            chunk = keys[start : start + _GET_LIMIT]
            resp = self._client.get_vectors(
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

    def delete_vectors(self, keys: list[str]) -> None:
        if not keys:
            return
        for start in range(0, len(keys), _PUT_LIMIT):
            chunk = keys[start : start + _PUT_LIMIT]
            self._client.delete_vectors(
                vectorBucketName=self._config.vector_bucket,
                indexName=self._config.index,
                keys=chunk,
            )
