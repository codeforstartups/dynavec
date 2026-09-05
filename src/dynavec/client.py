"""The Dynavec client: orchestrates DynamoDB + S3 Vectors into one vector DB.

Write path
----------
1. (optional) embed text with the configured, bring-your-own-key embedder
2. split metadata: small *filterable* subset -> S3 Vectors, full copy -> DynamoDB
3. ``put_vectors`` into S3 Vectors (key = "{namespace}#{id}")
4. ``batch_writer`` the full document + metadata into DynamoDB

Read path
---------
1. (optional) embed the query text
2. ``query_vectors`` in S3 Vectors -> keys + distances (AWS-managed ANN)
3. ``BatchGetItem`` in DynamoDB to hydrate full text + metadata (single-digit ms)
4. (optional) MMR rerank / caller-side RRF hybrid fusion
"""

from __future__ import annotations

from typing import Any, Optional, Union

from .config import DynavecConfig
from .embeddings.base import Embedder
from .exceptions import ConfigurationError, DimensionMismatchError
from .metadata import build_s3_filter, generate_auto_metadata, split_metadata
from .models import Document, SearchResult, UpsertResult
from .provisioning import provision_all
from .retrieval import distance_to_score, maximal_marginal_relevance
from .stores import DynamoDBStore, S3VectorsStore

Metadata = dict[str, Any]
_KEY_SEP = "#"


class Dynavec:
    """A serverless, in-your-own-account hybrid vector database."""

    def __init__(
        self,
        config: DynavecConfig,
        embedder: Optional[Embedder] = None,
        boto_session=None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._boto_session = boto_session
        self._vectors = S3VectorsStore(config, boto_session=boto_session)
        self._docs = DynamoDBStore(config, boto_session=boto_session)

        if embedder is not None and embedder.dimension != config.dimension:
            raise ConfigurationError(
                f"Embedder dimension ({embedder.dimension}) != index dimension "
                f"({config.dimension}). Fix the embedder or DynavecConfig.dimension."
            )

        if config.auto_provision:
            self.provision()

    # ------------------------------------------------------------------ setup
    def provision(self) -> None:
        """Create the S3 vector bucket, index, and DynamoDB table (idempotent)."""
        provision_all(self.config, boto_session=self._boto_session)

    # ------------------------------------------------------------- key helpers
    def _s3_key(self, namespace: str, doc_id: str) -> str:
        return f"{namespace}{_KEY_SEP}{doc_id}"

    def _split_key(self, key: str) -> tuple[str, str]:
        namespace, _, doc_id = key.partition(_KEY_SEP)
        return namespace, doc_id

    # --------------------------------------------------------------- write path
    def upsert(
        self,
        documents: Optional[list[Union[Document, dict]]] = None,
        *,
        namespace: str = "default",
        auto_metadata: bool = False,
    ) -> UpsertResult:
        """Insert or overwrite documents.

        Each item is a :class:`Document` (or an equivalent dict with ``id`` and
        one of ``text``/``vector``). If ``auto_metadata`` is True, dynavec
        derives lightweight metadata (timestamp, hash, counts) and merges it
        *under* any user-supplied metadata (user keys win).
        """
        if not documents:
            return UpsertResult(count=0, ids=[])

        docs: list[Document] = [
            d if isinstance(d, Document) else Document(**d) for d in documents
        ]

        # 1) embed any text-only documents in one batched call
        to_embed = [(i, d.text) for i, d in enumerate(docs) if d.vector is None]
        if to_embed:
            if self.embedder is None:
                raise ConfigurationError(
                    "Some documents have no vector and no embedder is configured. "
                    "Pass an embedder to Dynavec(...) or provide precomputed vectors."
                )
            vectors = self.embedder.embed_documents([t for _, t in to_embed])
            for (idx, _), vec in zip(to_embed, vectors):
                docs[idx].vector = vec

        # 2) validate dimensions + build store payloads
        s3_payload: list[tuple[str, list[float], Metadata]] = []
        ddb_payload: list[tuple[str, Optional[str], Metadata]] = []
        ids: list[str] = []

        for d in docs:
            if len(d.vector) != self.config.dimension:
                raise DimensionMismatchError(
                    f"Document {d.id!r} vector has dimension {len(d.vector)}, "
                    f"expected {self.config.dimension}."
                )
            meta = dict(d.metadata)
            if auto_metadata:
                auto = generate_auto_metadata(d.text)
                auto.update(meta)  # user-supplied keys take precedence
                meta = auto

            s3_meta, ddb_meta = split_metadata(meta, self.config, namespace, d.text)
            s3_payload.append((self._s3_key(namespace, d.id), d.vector, s3_meta))
            ddb_payload.append((d.id, d.text, ddb_meta))
            ids.append(d.id)

        # 3) write both stores
        self._vectors.put_vectors(s3_payload)
        self._docs.put_many(namespace, ddb_payload)

        return UpsertResult(count=len(ids), ids=ids)

    # ---------------------------------------------------------------- read path
    def search(
        self,
        query: Optional[str] = None,
        *,
        vector: Optional[list[float]] = None,
        top_k: int = 10,
        namespace: str = "default",
        filter: Optional[Metadata] = None,
        rerank: Optional[str] = None,  # None | "mmr"
        mmr_lambda: float = 0.5,
        include_vectors: bool = False,
    ) -> list[SearchResult]:
        """Semantic search. Provide ``query`` (embedded) or a raw ``vector``."""
        query_vector = self._resolve_query_vector(query, vector)

        # Over-fetch a larger candidate pool when we're going to rerank.
        fetch_k = top_k * self.config.over_fetch if rerank else top_k
        s3_filter = build_s3_filter(filter, namespace)

        raw = self._vectors.query(
            query_vector=query_vector,
            top_k=fetch_k,
            filter=s3_filter,
            return_metadata=True,
            return_distance=True,
        )

        # Map S3 hits -> (id, distance)
        hits: list[tuple[str, Optional[float]]] = []
        for v in raw:
            _, doc_id = self._split_key(v["key"])
            hits.append((doc_id, v.get("distance")))

        if not hits:
            return []

        # Hydrate documents from DynamoDB (single-digit-ms BatchGetItem)
        ids = [h[0] for h in hits]
        hydrated = self._docs.get_many(namespace, ids)

        # Optionally fetch stored vectors (needed for MMR / include_vectors)
        vec_by_key: dict[str, dict[str, Any]] = {}
        if rerank == "mmr" or include_vectors:
            keys = [self._s3_key(namespace, doc_id) for doc_id in ids]
            vec_by_key = self._vectors.get_vectors(keys)

        results: list[SearchResult] = []
        for doc_id, distance in hits:
            doc = hydrated.get(doc_id, {})
            score = distance_to_score(distance, self.config.distance_metric) if distance is not None else 0.0
            vec = None
            if vec_by_key:
                vec = vec_by_key.get(self._s3_key(namespace, doc_id), {}).get("vector")
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    distance=distance,
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                    vector=vec,
                )
            )

        if rerank == "mmr":
            results = maximal_marginal_relevance(query_vector, results, top_k, mmr_lambda)
        else:
            results = results[:top_k]

        if not include_vectors:
            for r in results:
                r.vector = None

        return results

    def _resolve_query_vector(
        self, query: Optional[str], vector: Optional[list[float]]
    ) -> list[float]:
        if vector is not None:
            if len(vector) != self.config.dimension:
                raise DimensionMismatchError(
                    f"Query vector dimension {len(vector)} != {self.config.dimension}."
                )
            return vector
        if query is None:
            raise ValueError("Provide either 'query' text or a 'vector'.")
        if self.embedder is None:
            raise ConfigurationError(
                "Text query requires an embedder. Pass one to Dynavec(...) or query "
                "with a precomputed 'vector'."
            )
        return self.embedder.embed_query(query)

    # ------------------------------------------------------------------- CRUD
    def get(self, ids: list[str], namespace: str = "default") -> list[SearchResult]:
        """Fetch documents by id (no search)."""
        hydrated = self._docs.get_many(namespace, ids)
        return [
            SearchResult(
                id=doc_id,
                score=1.0,
                text=hydrated[doc_id].get("text"),
                metadata=hydrated[doc_id].get("metadata", {}),
            )
            for doc_id in ids
            if doc_id in hydrated
        ]

    def delete(self, ids: list[str], namespace: str = "default") -> None:
        """Delete documents from both stores."""
        keys = [self._s3_key(namespace, doc_id) for doc_id in ids]
        self._vectors.delete_vectors(keys)
        self._docs.delete_many(namespace, ids)
