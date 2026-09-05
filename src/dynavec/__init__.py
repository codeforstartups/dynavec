"""dynavec — serverless hybrid vector database on DynamoDB + Amazon S3 Vectors.

Quick start
-----------
    from dynavec import Dynavec, DynavecConfig
    from dynavec.embeddings import OpenAIEmbedder

    cfg = DynavecConfig(
        vector_bucket="my-vectors",
        index="docs",
        table="dynavec_docs",
        dimension=1536,
        auto_provision=True,
    )
    db = Dynavec(cfg, embedder=OpenAIEmbedder(model="text-embedding-3-small"))

    db.upsert([{"id": "a", "text": "hello world", "metadata": {"lang": "en"}}])
    hits = db.search("greetings", top_k=3, filter={"lang": "en"})
"""

from __future__ import annotations

from .client import Dynavec
from .config import DynavecConfig
from .exceptions import (
    ConfigurationError,
    DimensionMismatchError,
    DynavecError,
    EmbeddingError,
    MissingDependencyError,
    NotFoundError,
    ProvisioningError,
)
from .models import Document, SearchResult, UpsertResult
from .retrieval import (
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)

__version__ = "0.1.0"

__all__ = [
    "Dynavec",
    "DynavecConfig",
    "Document",
    "SearchResult",
    "UpsertResult",
    "reciprocal_rank_fusion",
    "maximal_marginal_relevance",
    # exceptions
    "DynavecError",
    "ConfigurationError",
    "ProvisioningError",
    "EmbeddingError",
    "DimensionMismatchError",
    "NotFoundError",
    "MissingDependencyError",
]
