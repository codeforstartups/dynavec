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

from .cache import DynamoDBCache, RedisCache, SemanticCache
from .client import Dynavec
from .config import DynavecConfig
from .credentials import AWSCredentials
from .exceptions import (
    ConfigurationError,
    DimensionMismatchError,
    DynavecError,
    EmbeddingError,
    MissingDependencyError,
    NotFoundError,
    ProvisioningError,
)
from .graph import GraphStore
from .models import Document, SearchResult, UpsertResult
from .namespace import NamespaceView
from .quantization import ProductQuantizer
from .retrieval import (
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from .transforms import LambdaTransform, TransformContext, TransformPipeline

__version__ = "0.2.0"

__all__ = [
    "Dynavec",
    "DynavecConfig",
    "AWSCredentials",
    "Document",
    "SearchResult",
    "UpsertResult",
    "NamespaceView",
    "ProductQuantizer",
    "GraphStore",
    "SemanticCache",
    "DynamoDBCache",
    "RedisCache",
    "reciprocal_rank_fusion",
    "maximal_marginal_relevance",
    "TransformPipeline",
    "TransformContext",
    "LambdaTransform",
    # exceptions
    "DynavecError",
    "ConfigurationError",
    "ProvisioningError",
    "EmbeddingError",
    "DimensionMismatchError",
    "NotFoundError",
    "MissingDependencyError",
]
