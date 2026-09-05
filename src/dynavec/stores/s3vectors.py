"""Amazon S3 Vectors store: the serverless, billion-scale ANN tier.

Wraps the boto3 ``s3vectors`` client. S3 Vectors owns the approximate-nearest-
neighbor index (AWS-managed, ~90%+ recall, ~100ms warm / sub-second cold). We
store the vector + a *small filterable* metadata subset here; the full document
lives in DynamoDB.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..config import DynavecConfig

Metadata = dict[str, Any]

# PutVectors accepts up to 500 vectors per call.
_PUT_LIMIT = 500
_GET_LIMIT = 100


def _f32(vector: list[float]) -> list[float]:
    """Ensure vector is float32-clean (S3 Vectors stores float32)."""
    return np.asarray(vector, dtype=np.float32).tolist()


class S3VectorsStore:
    def __init__(self, config: DynavecConfig, boto_session=None) -> None:
        import boto3  # local import: base import stays cheap

        session = boto_session or boto3.Session()
        self._config = config
        self._client = session.client("s3vectors", region_name=config.region)

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
            self._client.put_vectors(
                vectorBucketName=self._config.vector_bucket,
                indexName=self._config.index,
                vectors=payload,
            )

    def query(
        self,
        query_vector: list[float],
        top_k: int,
        filter: Optional[Metadata] = None,
        return_metadata: bool = True,
        return_distance: bool = True,
    ) -> list[dict[str, Any]]:
        """Run an ANN query. Returns a list of ``{key, distance?, metadata?}``."""
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
        resp = self._client.query_vectors(**kwargs)
        return resp.get("vectors", [])

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
