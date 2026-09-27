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

from .cache import BaseCache, DynamoDBCache, RedisCache, SemanticCache, warm_cache
from .client import Dynavec
from .config import DynavecConfig
from .credentials import AWSCredentials
from .exceptions import (
    ConfigurationError,
    ConflictError,
    DimensionMismatchError,
    DynavecError,
    EmbeddingError,
    ItemTooLargeError,
    MissingDependencyError,
    NotFoundError,
    ProvisioningError,
)
from .fusion import FitResult, RRFWeightFitter
from .graph import GraphStore
from .hot import HotTier
from .ingest import (
    CsvSource,
    DocxSource,
    IterableSource,
    MarkdownSource,
    MCPResourceSource,
    PDFSource,
    PptxSource,
    Record,
    S3Source,
    URLSource,
    XlsxSource,
    ingest,
)
from .models import (
    Document,
    ExplainedSearchResult,
    IndexInfo,
    SearchExplanation,
    SearchResult,
    UpsertResult,
)
from .namespace import NamespaceView
from .quantization import (
    OPQRotation,
    OptimizedProductQuantizer,
    ProductQuantizer,
    ScalarQuantizer,
)
from .retrieval import (
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from .retrievers import (
    HyDERetriever,
    MultiQueryRetriever,
    QueryExpansionRetriever,
)
from .spfresh import (
    Partition,
    SPFreshConfig,
    SPFreshHotIndex,
    SPFreshRebalancer,
)
from .transforms import LambdaTransform, TransformContext, TransformPipeline

__version__ = "0.6.0"

__all__ = [
    "Dynavec",
    "DynavecConfig",
    "AWSCredentials",
    "Document",
    "IndexInfo",
    "SearchResult",
    "SearchExplanation",
    "ExplainedSearchResult",
    "UpsertResult",
    "NamespaceView",
    "ProductQuantizer",
    "ScalarQuantizer",
    "OPQRotation",
    "OptimizedProductQuantizer",
    "GraphStore",
    "BaseCache",
    "SemanticCache",
    "DynamoDBCache",
    "RedisCache",
    "warm_cache",
    "reciprocal_rank_fusion",
    "maximal_marginal_relevance",
    "QueryExpansionRetriever",
    "MultiQueryRetriever",
    "HyDERetriever",
    "FitResult",
    "RRFWeightFitter",
    "HotTier",
    "Partition",
    "SPFreshConfig",
    "SPFreshHotIndex",
    "SPFreshRebalancer",
    "TransformPipeline",
    "TransformContext",
    "LambdaTransform",
    # Ingestion
    "Record",
    "ingest",
    "IterableSource",
    "PDFSource",
    "CsvSource",
    "DocxSource",
    "PptxSource",
    "XlsxSource",
    "URLSource",
    "MarkdownSource",
    "MCPResourceSource",
    "S3Source",
    # exceptions
    "DynavecError",
    "ConfigurationError",
    "ProvisioningError",
    "EmbeddingError",
    "DimensionMismatchError",
    "NotFoundError",
    "ItemTooLargeError",
    "ConflictError",
    "MissingDependencyError",
]
