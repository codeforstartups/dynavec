"""Configuration objects for dynavec.

Everything the client needs to talk to *the user's own* AWS account lives here.
No data leaves the account: DynamoDB + S3 Vectors are both regional AWS services
and dynavec only ever calls them with the caller's credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

DistanceMetric = Literal["cosine", "euclidean"]

# S3 Vectors reserves a couple of metadata keys for dynavec's own bookkeeping.
NS_METADATA_KEY = "_dv_ns"
TEXT_METADATA_KEY = "_dv_text"  # optional truncated text mirror (non-filterable)


@dataclass(frozen=True)
class DynavecConfig:
    """Top-level configuration for a :class:`~dynavec.client.Dynavec` client.

    Parameters
    ----------
    vector_bucket:
        Name of the S3 vector bucket (created if ``auto_provision=True``).
    index:
        Name of the vector index within the bucket.
    table:
        DynamoDB table name for the document / metadata store.
    dimension:
        Embedding dimension. Must match the embedder and the S3 Vectors index.
    distance_metric:
        ``"cosine"`` (default) or ``"euclidean"``.
    region:
        AWS region. Falls back to the standard boto3 resolution if ``None``.
    filterable_keys:
        Metadata keys that should be pushed into S3 Vectors so they can be used
        as pre-filters during ANN search. Everything else lives only in
        DynamoDB. Keep this list small — S3 Vectors caps filterable metadata size
        per vector. ``None`` (default) means "push all metadata keys as
        filterable" (fine for small metadata; not recommended with large text).
    store_text_in_s3vectors:
        If True, mirror a truncated copy of the source text into S3 Vectors as
        *non-filterable* metadata. Off by default — the canonical text lives in
        DynamoDB, which is cheaper to read and has no per-vector size cap.
    over_fetch:
        Multiplier applied to ``top_k`` when reranking is enabled, so the
        reranker has a candidate pool to work with.
    """

    vector_bucket: str
    index: str
    table: str
    dimension: int
    distance_metric: DistanceMetric = "cosine"
    region: Optional[str] = None

    # metadata split between the two stores
    filterable_keys: Optional[list[str]] = None
    non_filterable_keys: list[str] = field(default_factory=list)
    store_text_in_s3vectors: bool = False
    text_mirror_max_chars: int = 2048

    # retrieval tuning
    over_fetch: int = 4

    # concurrency (I/O-bound: threads give real parallelism as boto3 releases
    # the GIL during network calls). See client._executor.
    max_workers: int = 8
    parallel_writes: bool = True

    # provisioning
    auto_provision: bool = False
    dynamodb_billing_mode: Literal["PAY_PER_REQUEST", "PROVISIONED"] = "PAY_PER_REQUEST"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if self.distance_metric not in ("cosine", "euclidean"):
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        if self.over_fetch < 1:
            raise ValueError("over_fetch must be >= 1")
