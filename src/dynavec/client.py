"""The Dynavec client: orchestrates DynamoDB + S3 Vectors into one vector DB.

Write path
----------
1. run transform pipeline (enrich / redact / Lambda) over each document
2. (optional) embed text with the configured, bring-your-own-key embedder
3. split metadata: small *filterable* subset -> S3 Vectors, full copy -> DynamoDB
4. ``put_vectors`` into S3 Vectors + ``batch_writer`` into DynamoDB (in parallel)

Read path
---------
1. (optional) embed the query text
2. ``query_vectors`` in S3 Vectors -> keys + distances (AWS-managed ANN)
3. ``BatchGetItem`` in DynamoDB to hydrate full text + metadata (single-digit ms)
4. (optional) client-side rescore (cosine/dot/euclidean/manhattan/combination),
   MMR rerank, or streaming page-by-page delivery to the agent

Concurrency
-----------
The workload is I/O-bound (network calls to AWS). Python's GIL is released
during those calls, so a ``ThreadPoolExecutor`` gives real parallelism for
batched writes and multi-query reads without the complexity of a full async
rewrite. (A native asyncio client is on the roadmap.)
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .cache import BaseCache

import numpy as np

from .config import DynavecConfig
from .credentials import AWSCredentials, resolve_session
from .embeddings.base import Embedder
from .exceptions import ConfigurationError, DimensionMismatchError, NotFoundError
from .graph import GraphStore
from .metadata import build_s3_filter, generate_auto_metadata, split_metadata
from .metrics import normalize_scores as normalize_metric_scores
from .metrics import rescore as metric_rescore
from .metrics import score as metric_score
from .models import Document, SearchResult, UpsertResult
from .namespace import NamespaceView
from .provisioning import provision_all
from .retrieval import distance_to_score, maximal_marginal_relevance
from .stores import DynamoDBStore, S3VectorsStore
from .transforms import TransformContext, as_pipeline
from .utils import chunked

Metadata = dict[str, Any]
_KEY_SEP = "#"
_S3_PUT_CHUNK = 500
_DDB_CHUNK = 500

RescoreSpec = Union[str, dict]  # "cosine" | "manhattan" | {"cosine":0.7,"dot":0.3}


class Dynavec:
    """A serverless, in-your-own-account hybrid vector database."""

    def __init__(
        self,
        config: DynavecConfig,
        embedder: Embedder | None = None,
        *,
        credentials: AWSCredentials | None = None,
        boto_session=None,
        transform=None,
        cache=None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._session = resolve_session(credentials, boto_session)
        self._vectors = S3VectorsStore(config, boto_session=self._session)
        self._docs = DynamoDBStore(config, boto_session=self._session)
        self._default_transform = as_pipeline(transform)
        self._cache = cache
        self._graph_store: GraphStore | None = None
        self._pool: ThreadPoolExecutor | None = None

        if embedder is not None and embedder.dimension != config.dimension:
            raise ConfigurationError(
                f"Embedder dimension ({embedder.dimension}) != index dimension "
                f"({config.dimension}). Fix the embedder or DynavecConfig.dimension."
            )

        if config.auto_provision:
            self.provision()

    # ------------------------------------------------------------ lifecycle
    @property
    def cache(self) -> BaseCache | None:
        """The configured query cache, if any."""
        return self._cache

    @property
    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> Dynavec:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ setup
    def provision(self) -> None:
        """Create the S3 vector bucket, index, and DynamoDB table (idempotent)."""
        provision_all(self.config, boto_session=self._session)

    def namespace(self, namespace: str) -> NamespaceView:
        """Return a handle with every op bound to ``namespace`` (namespace RAG)."""
        return NamespaceView(self, namespace)

    @property
    def graph(self) -> GraphStore:
        """Lazily-built knowledge-graph store (shares the DynamoDB table)."""
        if self._graph_store is None:
            self._graph_store = GraphStore(self.config, boto_session=self._session)
        return self._graph_store

    # ------------------------------------------------------------- key helpers
    def _s3_key(self, namespace: str, doc_id: str) -> str:
        return f"{namespace}{_KEY_SEP}{doc_id}"

    def _split_key(self, key: str) -> tuple[str, str]:
        namespace, _, doc_id = key.partition(_KEY_SEP)
        return namespace, doc_id

    def _run_parallel(self, tasks: list) -> None:
        """Run zero-arg callables; parallel if enabled, else sequential."""
        if not tasks:
            return
        if not self.config.parallel_writes or len(tasks) == 1:
            for t in tasks:
                t()
            return
        futures = [self._executor.submit(t) for t in tasks]
        for f in futures:
            f.result()  # propagate the first exception

    # --------------------------------------------------------------- write path
    def _prepare(
        self,
        docs: list[Document],
        namespace: str,
        auto_metadata: bool,
        transform,
    ) -> tuple[list[tuple], list[tuple], list[str]]:
        pipeline = as_pipeline(transform) or self._default_transform

        # 1) transforms may set/rewrite text, vector, metadata
        if pipeline is not None:
            for d in docs:
                ctx = pipeline(
                    TransformContext(
                        id=d.id, text=d.text, vector=d.vector,
                        metadata=dict(d.metadata), namespace=namespace,
                    )
                )
                d.text, d.vector, d.metadata = ctx.text, ctx.vector, ctx.metadata

        # 2) embed anything still missing a vector, in one batched call
        to_embed = [(i, d.text) for i, d in enumerate(docs) if d.vector is None]
        if to_embed:
            if self.embedder is None:
                raise ConfigurationError(
                    "Some documents have no vector and no embedder is configured. "
                    "Pass an embedder to Dynavec(...) or provide precomputed vectors."
                )
            texts = [t for _, t in to_embed]
            if any(t is None for t in texts):
                raise ConfigurationError("A document has neither text nor vector.")
            vectors = self.embedder.embed_documents(texts)
            for (idx, _), vec in zip(to_embed, vectors):
                docs[idx].vector = vec

        # 3) validate + build payloads
        s3_payload, ddb_payload, ids = [], [], []
        for d in docs:
            if len(d.vector) != self.config.dimension:
                raise DimensionMismatchError(
                    f"Document {d.id!r} vector has dimension {len(d.vector)}, "
                    f"expected {self.config.dimension}."
                )
            meta = dict(d.metadata)
            if auto_metadata:
                auto = generate_auto_metadata(d.text)
                auto.update(meta)
                meta = auto
            s3_meta, ddb_meta = split_metadata(meta, self.config, namespace, d.text)
            s3_payload.append((self._s3_key(namespace, d.id), d.vector, s3_meta))
            ddb_payload.append((d.id, d.text, ddb_meta))
            ids.append(d.id)
        return s3_payload, ddb_payload, ids

    def _write(self, namespace: str, s3_payload: list, ddb_payload: list) -> None:
        tasks = []
        for chunk in chunked(s3_payload, _S3_PUT_CHUNK):
            tasks.append(lambda c=chunk: self._vectors.put_vectors(c))
        for chunk in chunked(ddb_payload, _DDB_CHUNK):
            tasks.append(lambda c=chunk: self._docs.put_many(namespace, c))
        self._run_parallel(tasks)

    def upsert(
        self,
        documents: list[Document | dict] | None = None,
        *,
        namespace: str = "default",
        auto_metadata: bool = False,
        transform=None,
    ) -> UpsertResult:
        """Insert or overwrite documents (each a :class:`Document` or dict)."""
        if not documents:
            return UpsertResult(count=0, ids=[])
        docs = [d if isinstance(d, Document) else Document(**d) for d in documents]
        s3_payload, ddb_payload, ids = self._prepare(docs, namespace, auto_metadata, transform)
        self._write(namespace, s3_payload, ddb_payload)
        return UpsertResult(count=len(ids), ids=ids)

    def update(
        self,
        id: str,
        *,
        namespace: str = "default",
        text: str | None = None,
        vector: list[float] | None = None,
        metadata: Metadata | None = None,
        merge_metadata: bool = True,
        transform=None,
        upsert_if_missing: bool = False,
    ) -> UpsertResult:
        """Update an existing document's text, vector, and/or metadata.

        Read-modify-write: metadata is merged by default; the vector is re-derived
        only when text changes (and an embedder exists) or a new vector is given,
        otherwise the stored vector is preserved.
        """
        existing = self._docs.get_many(namespace, [id]).get(id)
        if existing is None and not upsert_if_missing:
            raise NotFoundError(f"Document {id!r} not found in namespace {namespace!r}.")
        existing = existing or {"text": None, "metadata": {}}

        new_text = text if text is not None else existing.get("text")

        # decide the vector
        new_vector = vector
        if new_vector is None:
            if text is not None and self.embedder is not None:
                new_vector = self.embedder.embed_documents([new_text])[0]
            else:
                fetched = self._vectors.get_vectors([self._s3_key(namespace, id)])
                got = fetched.get(self._s3_key(namespace, id))
                if got and got.get("vector") is not None:
                    new_vector = got["vector"]
        if new_vector is None:
            raise ConfigurationError(
                f"Cannot resolve a vector for update of {id!r}: pass 'vector', or "
                "'text' with an embedder configured."
            )

        # decide metadata
        if metadata is None:
            new_meta = existing.get("metadata", {})
        elif merge_metadata:
            new_meta = {**existing.get("metadata", {}), **metadata}
        else:
            new_meta = metadata

        doc = Document(id=id, text=new_text, vector=new_vector, metadata=new_meta)
        # mark op=update for any transform that cares
        s3_payload, ddb_payload, ids = self._prepare(
            [doc], namespace, auto_metadata=False, transform=transform
        )
        self._write(namespace, s3_payload, ddb_payload)
        return UpsertResult(count=1, ids=ids)

    # ---------------------------------------------------------------- read path
    def search(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 10,
        namespace: str = "default",
        filter: Metadata | None = None,
        rescore: RescoreSpec | None = None,
        rerank: str | None = None,  # None | "mmr"
        mmr_lambda: float = 0.5,
        include_vectors: bool = False,
        use_cache: bool | None = None,
        normalize_scores: bool = False,
    ) -> list[SearchResult]:
        """Semantic search. Provide ``query`` (embedded) or a raw ``vector``.

        ``rescore`` re-orders the ANN candidates with a client-side metric
        (``"cosine"``, ``"dot"``, ``"euclidean"``, ``"manhattan"``) or a weighted
        combination like ``{"cosine": 0.7, "manhattan": 0.3}``.
        Set ``normalize_scores=True`` to min-max normalize the final result set
        to ``[0, 1]`` without changing its order.

        If a cache is configured, repeated/similar queries are served from it
        (set ``use_cache=False`` to force a fresh search).
        """
        query_vector = self._resolve_query_vector(query, vector)

        # cache key includes ranking options so different ranking != same entry
        cache_on = self._cache is not None if use_cache is None else use_cache
        cache_filter = None
        if cache_on and self._cache is not None:
            cache_filter = {
                **(filter or {}),
                "__rank": {
                    "rescore": rescore,
                    "rerank": rerank,
                    "mmr": mmr_lambda,
                    "normalize_scores": normalize_scores,
                },
            }
            cached = self._cache.get(namespace, query_vector, top_k, cache_filter)
            if cached is not None:
                return cached

        needs_vectors = rerank == "mmr" or rescore is not None or include_vectors
        fetch_k = top_k * self.config.over_fetch if (rerank or rescore) else top_k

        raw = self._vectors.query(
            query_vector=query_vector,
            top_k=fetch_k,
            filter=build_s3_filter(filter, namespace),
            return_metadata=True,
            return_distance=True,
        )
        if not raw:
            return []

        hits = [(self._split_key(v["key"])[1], v.get("distance")) for v in raw]
        ids = [h[0] for h in hits]
        hydrated = self._docs.get_many(namespace, ids)

        vec_by_key = {}
        if needs_vectors:
            vec_by_key = self._vectors.get_vectors(
                [self._s3_key(namespace, doc_id) for doc_id in ids]
            )

        results: list[SearchResult] = []
        for doc_id, distance in hits:
            doc = hydrated.get(doc_id, {})
            vec = vec_by_key.get(self._s3_key(namespace, doc_id), {}).get("vector") if vec_by_key else None
            results.append(
                SearchResult(
                    id=doc_id,
                    score=distance_to_score(distance, self.config.distance_metric)
                    if distance is not None else 0.0,
                    distance=distance,
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                    vector=vec,
                )
            )

        if rescore is not None:
            results = self._apply_rescore(query_vector, results, rescore)
        if rerank == "mmr":
            results = maximal_marginal_relevance(query_vector, results, top_k, mmr_lambda)
        else:
            results = results[:top_k]

        if normalize_scores and results:
            normalized = normalize_metric_scores(np.asarray([r.score for r in results]))
            for result, normalized_score in zip(results, normalized):
                result.score = float(normalized_score)

        if not include_vectors:
            for r in results:
                r.vector = None

        if cache_on and self._cache is not None and results:
            self._cache.put(namespace, query_vector, top_k, cache_filter, results)
        return results

    def _apply_rescore(
        self, query_vector: list[float], results: list[SearchResult], spec: RescoreSpec
    ) -> list[SearchResult]:
        scored = [r for r in results if r.vector is not None]
        if not scored:
            return results
        mat = np.asarray([r.vector for r in scored], dtype=np.float32)
        order, scores = metric_rescore(np.asarray(query_vector, dtype=np.float32), mat, spec)
        out = []
        for rank_pos in order:
            r = scored[int(rank_pos)]
            r.score = float(scores[int(rank_pos)])
            out.append(r)
        return out

    def search_stream(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 50,
        namespace: str = "default",
        filter: Metadata | None = None,
    ) -> Iterator[SearchResult]:
        """Stream results to the agent page-by-page as S3 Vectors returns them.

        A generator: the caller (agent) can start consuming the first hits before
        the full result set is retrieved. Reranking/rescoring are not applied in
        streaming mode (they need the whole candidate set).
        """
        query_vector = self._resolve_query_vector(query, vector)
        yielded = 0
        for page in self._vectors.query_pages(
            query_vector=query_vector,
            top_k=top_k,
            filter=build_s3_filter(filter, namespace),
            return_metadata=True,
            return_distance=True,
        ):
            page_hits = [(self._split_key(v["key"])[1], v.get("distance")) for v in page]
            hydrated = self._docs.get_many(namespace, [h[0] for h in page_hits])
            for doc_id, distance in page_hits:
                if yielded >= top_k:
                    return
                doc = hydrated.get(doc_id, {})
                yield SearchResult(
                    id=doc_id,
                    score=distance_to_score(distance, self.config.distance_metric)
                    if distance is not None else 0.0,
                    distance=distance,
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                )
                yielded += 1

    def search_many(
        self, queries: list[str], *, top_k: int = 10, namespace: str = "default", **kw
    ) -> list[list[SearchResult]]:
        """Run several queries concurrently (thread pool over I/O-bound calls)."""
        futures = [
            self._executor.submit(self.search, q, top_k=top_k, namespace=namespace, **kw)
            for q in queries
        ]
        return [f.result() for f in futures]

    def _resolve_query_vector(
        self, query: str | None, vector: list[float] | None
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

    # -------------------------------------------------------------- graph / ER
    def graph_add_node(self, entity_id, *, namespace="default", ntype=None, props=None):
        """Create/update a graph entity (a 'meaning' node)."""
        self.graph.add_node(namespace, entity_id, ntype, props)

    def graph_add_edge(self, src, relation, dst, *, namespace="default", bidirectional=False):
        """Relate two entities: ``(src) -[relation]-> (dst)``."""
        self.graph.add_edge(namespace, src, relation, dst)
        if bidirectional:
            self.graph.add_edge(namespace, dst, relation, src)

    def graph_link(self, entity_id, doc_ids, *, namespace="default"):
        """Attach documents (their S3 Vectors embeddings) to an entity."""
        self.graph.link_docs(namespace, entity_id, list(doc_ids))

    def graph_neighbors(self, entity_id, *, namespace="default", relation=None, hops=1):
        """Breadth-first traversal returning reachable entity ids (excl. seed)."""
        visited = {entity_id}
        frontier = [entity_id]
        for _ in range(hops):
            nxt = []
            for node in frontier:
                for nb in self.graph.neighbors(namespace, node, relation):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break
        return [e for e in visited if e != entity_id]

    def graph_search(
        self,
        query: str | None = None,
        *,
        seed_entities: list[str],
        vector: list[float] | None = None,
        namespace: str = "default",
        relation: str | None = None,
        hops: int = 1,
        top_k: int = 10,
        metric: str = "cosine",
    ) -> list[SearchResult]:
        """GraphRAG: traverse the graph from ``seed_entities`` to a candidate doc
        set, then rank those docs against the query embedding.

        The graph *narrows* the search — instead of ANN over everything, we score
        only structurally-related documents, which is both faster and more precise
        when relationships matter.
        """
        query_vector = self._resolve_query_vector(query, vector)

        entities = list(dict.fromkeys(seed_entities))
        for seed in seed_entities:
            entities += self.graph_neighbors(
                seed, namespace=namespace, relation=relation, hops=hops
            )
        entities = list(dict.fromkeys(entities))

        doc_ids = self.graph.get_docs(namespace, entities)
        if not doc_ids:
            return []

        vec_by_key = self._vectors.get_vectors(
            [self._s3_key(namespace, d) for d in doc_ids]
        )
        hydrated = self._docs.get_many(namespace, doc_ids)

        scored = [d for d in doc_ids if vec_by_key.get(self._s3_key(namespace, d), {}).get("vector")]
        if not scored:
            return []
        mat = np.asarray(
            [vec_by_key[self._s3_key(namespace, d)]["vector"] for d in scored],
            dtype=np.float32,
        )
        scores = metric_score(np.asarray(query_vector, dtype=np.float32), mat, metric)
        order = np.argsort(-scores)[:top_k]

        out = []
        for i in order:
            d = scored[int(i)]
            doc = hydrated.get(d, {})
            out.append(
                SearchResult(
                    id=d, score=float(scores[int(i)]),
                    text=doc.get("text"), metadata=doc.get("metadata", {}),
                )
            )
        return out

    def delete(self, ids: list[str], namespace: str = "default") -> None:
        """Delete documents from both stores."""
        keys = [self._s3_key(namespace, doc_id) for doc_id in ids]
        self._run_parallel([
            lambda: self._vectors.delete_vectors(keys),
            lambda: self._docs.delete_many(namespace, ids),
        ])
