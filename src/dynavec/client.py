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

import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, Union

if TYPE_CHECKING:
    from .cache import BaseCache

import numpy as np

from .config import NS_METADATA_KEY, TEXT_METADATA_KEY, DynavecConfig
from .credentials import AWSCredentials, resolve_session
from .embeddings.base import Embedder
from .exceptions import (
    ConfigurationError,
    ConflictError,
    DimensionMismatchError,
    MissingDependencyError,
    NotFoundError,
)
from .graph import GraphStore
from .hot import HotTier
from .metadata import build_s3_filter, generate_auto_metadata, split_metadata
from .metrics import normalize_scores as normalize_metric_scores
from .metrics import rescore as metric_rescore
from .metrics import score as metric_score
from .models import (
    Document,
    ExplainedSearchResult,
    IndexInfo,
    SearchExplanation,
    SearchResult,
    UpsertResult,
)
from .namespace import NamespaceView
from .provisioning import provision_all
from .retrieval import distance_to_score, maximal_marginal_relevance, reciprocal_rank_fusion
from .stores import DynamoDBStore, S3VectorsStore
from .stores.dynamodb import check_item_size
from .transforms import TransformContext, as_pipeline
from .utils import KEY_SEPARATOR, chunked, decode_key_component, encode_key_component

Metadata = dict[str, Any]
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
        telemetry=None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._session = resolve_session(credentials, boto_session)
        self._vectors = S3VectorsStore(config, boto_session=self._session)
        self._docs = DynamoDBStore(config, boto_session=self._session)
        self._default_transform = as_pipeline(transform)
        self._cache = cache
        self._telemetry = telemetry
        self._graph_store: GraphStore | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._hot: HotTier | None = HotTier(config) if config.hot_tier else None
        self._cross_encoder = None

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

    def describe(self) -> IndexInfo:
        """Return the live bucket/index/table config, as provisioned in AWS."""
        idx = self._vectors.get_index()["index"]
        table_desc = self._docs._ddb.meta.client.describe_table(TableName=self.config.table)[
            "Table"
        ]
        return IndexInfo(
            vector_bucket=self.config.vector_bucket,
            index=self.config.index,
            dimension=idx["dimension"],
            distance_metric=idx["distanceMetric"],
            table=self.config.table,
            table_status=table_desc["TableStatus"],
            non_filterable_keys=idx.get("metadataConfiguration", {}).get(
                "nonFilterableMetadataKeys", []
            ),
            item_count=table_desc.get("ItemCount"),
        )

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
        return f"{encode_key_component(namespace)}{KEY_SEPARATOR}{encode_key_component(doc_id)}"

    def _split_key(self, key: str) -> tuple[str, str]:
        namespace, _, doc_id = key.partition(KEY_SEPARATOR)
        return decode_key_component(namespace), decode_key_component(doc_id)

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
    ) -> tuple[list[tuple], list[tuple], list[str], list[tuple]]:
        pipeline = as_pipeline(transform) or self._default_transform

        # 1) transforms may set/rewrite text, vector, metadata
        if pipeline is not None:
            for d in docs:
                ctx = pipeline(
                    TransformContext(
                        id=d.id,
                        text=d.text,
                        vector=d.vector,
                        metadata=dict(d.metadata),
                        namespace=namespace,
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
        s3_payload, ddb_payload, ids, hot_payload = [], [], [], []
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
            # fail before either store is written, not partway through a batch
            check_item_size(namespace, d.id, d.text, ddb_meta, self.config.gzip_threshold_bytes)
            s3_payload.append((self._s3_key(namespace, d.id), d.vector, s3_meta))
            ddb_payload.append((d.id, d.text, ddb_meta))
            ids.append(d.id)
            # Hot tier keeps the full (merged) metadata + text so warmed
            # namespaces need neither an S3 query nor a DynamoDB read.
            hot_payload.append((d.id, d.vector, d.text, meta))
        return s3_payload, ddb_payload, ids, hot_payload

    def _write(self, namespace: str, s3_payload: list, ddb_payload: list) -> None:
        tasks = []
        for chunk in chunked(s3_payload, _S3_PUT_CHUNK):
            tasks.append(lambda c=chunk: self._vectors.put_vectors(c))
        for chunk in chunked(ddb_payload, _DDB_CHUNK):
            tasks.append(lambda c=chunk: self._docs.put_many(namespace, c))
        self._run_parallel(tasks)

    def _invalidate_cache(self, namespace: str) -> None:
        if self._cache is not None and self.config.cache_invalidate_on_write:
            self._cache.invalidate(namespace)

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
        s3_payload, ddb_payload, ids, hot_payload = self._prepare(
            docs, namespace, auto_metadata, transform
        )
        self._write(namespace, s3_payload, ddb_payload)
        if self._hot is not None:
            self._hot.insert_many(namespace, hot_payload)
        self._invalidate_cache(namespace)
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
        expected_version: int | None = None,
    ) -> UpsertResult:
        """Update an existing document's text, vector, and/or metadata.

        Read-modify-write: metadata is merged by default; the vector is re-derived
        only when text changes (and an embedder exists) or a new vector is given,
        otherwise the stored vector is preserved.

        The write is conditional on the document's version, so a concurrent
        update is never silently overwritten: if the document changed since it
        was read, :class:`ConflictError` is raised and nothing is written. Pass
        ``expected_version`` (the ``version`` from an earlier update's result)
        to also detect changes made since *your* last read. The returned
        :class:`UpsertResult` carries the new ``version``.
        """
        existing = self._docs.get_versioned(namespace, id)
        if existing is None and not upsert_if_missing:
            raise NotFoundError(f"Document {id!r} not found in namespace {namespace!r}.")
        existing = existing or {"text": None, "metadata": {}, "version": 0}
        if expected_version is not None and existing["version"] != expected_version:
            raise ConflictError(id, namespace, expected_version)

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
        s3_payload, ddb_payload, ids, hot_payload = self._prepare(
            [doc], namespace, auto_metadata=False, transform=transform
        )
        # The conditional DynamoDB write goes first: on a conflict it raises
        # before S3 Vectors or the hot tier are touched.
        (_, ddb_text, ddb_meta), = ddb_payload
        version = self._docs.put_versioned(
            namespace, id, ddb_text, ddb_meta, expected_version=existing["version"]
        )
        self._vectors.put_vectors(s3_payload)
        if self._hot is not None:
            self._hot.insert_many(namespace, hot_payload)
        self._invalidate_cache(namespace)
        return UpsertResult(count=1, ids=ids, version=version)

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
        explain: bool = False,
    ) -> list[SearchResult] | ExplainedSearchResult:
        """Semantic search. Provide ``query`` (embedded) or a raw ``vector``.

        ``rescore`` re-orders the ANN candidates with a client-side metric
        (``"cosine"``, ``"dot"``, ``"euclidean"``, ``"manhattan"``) or a weighted
        combination like ``{"cosine": 0.7, "manhattan": 0.3}``.
        Set ``normalize_scores=True`` to min-max normalize the final result set
        to ``[0, 1]`` without changing its order.

        If a cache is configured, repeated/similar queries are served from it
        (set ``use_cache=False`` to force a fresh search).
        Set ``explain=True`` to return results together with per-stage timings
        and candidate counts for debugging.
        """
        t0 = time.perf_counter()
        tel = self._telemetry
        explanation = SearchExplanation() if explain else None

        try:
            stage_t0 = time.perf_counter()
            query_vector = self._resolve_query_vector(query, vector)
            if explanation is not None:
                explanation.timings_ms["query_vector"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )

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
                stage_t0 = time.perf_counter()
                cached = self._cache.get(namespace, query_vector, top_k, cache_filter)

                if explanation is not None:
                    explanation.timings_ms["cache_lookup"] = round(
                        (time.perf_counter() - stage_t0) * 1000, 3
                    )
                    explanation.candidate_counts["cached"] = (
                        len(cached) if cached is not None else 0
                    )

                if cached is not None:
                    if explanation is not None:
                        explanation.candidate_counts["final"] = len(cached)
                        explanation.timings_ms["total"] = round(
                            (time.perf_counter() - t0) * 1000, 3
                        )

                    self._record_search(
                        tel,
                        t0,
                        namespace,
                        top_k,
                        cached,
                        True,
                        filter,
                        rescore,
                        rerank,
                        query,
                    )

                    if explanation is not None:
                        return ExplainedSearchResult(
                            results=cached,
                            explanation=explanation,
                        )

                    return cached

            results = self._search_core(
                query_vector,
                query=query,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                rescore=rescore,
                rerank=rerank,
                mmr_lambda=mmr_lambda,
                include_vectors=include_vectors,
                normalize_scores=normalize_scores,
                explanation=explanation,
            )

            if cache_on and self._cache is not None and results:
                stage_t0 = time.perf_counter()
                self._cache.put(namespace, query_vector, top_k, cache_filter, results)

                if explanation is not None:
                    explanation.timings_ms["cache_write"] = round(
                        (time.perf_counter() - stage_t0) * 1000, 3
                    )

            if explanation is not None:
                explanation.candidate_counts["final"] = len(results)
                explanation.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 3)

            self._record_search(
                tel,
                t0,
                namespace,
                top_k,
                results,
                (False if cache_on else None),
                filter,
                rescore,
                rerank,
                query,
            )

            if explanation is not None:
                return ExplainedSearchResult(
                    results=results,
                    explanation=explanation,
                )

            return results
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            if tel is not None:
                tel.record(
                    tel.new_event(
                        "search",
                        namespace=namespace,
                        top_k=top_k,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                        status="error",
                        error=str(exc)[:200],
                        filtered=bool(filter),
                        rescore=self._rescore_label(rescore),
                        rerank=rerank,
                    )
                )
            raise

    def _search_core(
        self,
        query_vector,
        *,
        query: str | None = None,
        top_k,
        namespace,
        filter,
        rescore,
        rerank,
        mmr_lambda,
        include_vectors,
        normalize_scores,
        explanation: SearchExplanation | None = None,
    ) -> list[SearchResult]:
        needs_vectors = rerank == "mmr" or rescore is not None or include_vectors
        fetch_k = top_k * self.config.over_fetch if (rerank or rescore) else top_k

        # Fast path: a warmed (authoritative) namespace is served entirely from
        # RAM — no S3 Vectors query and no DynamoDB hydration. Returns None when
        # the namespace isn't authoritative or the filter is unsupported, so we
        # transparently fall back to S3 (hot tier can only speed up, never break).
        results: list[SearchResult] | None = None
        if self._hot is not None:
            stage_t0 = time.perf_counter()
            results = self._hot.search(namespace, query_vector, fetch_k, filter)

            if explanation is not None:
                explanation.timings_ms["hot_lookup"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )
                if results is not None:
                    explanation.candidate_counts["retrieved"] = len(results)

        if results is None:
            stage_t0 = time.perf_counter()
            raw = self._vectors.query(
                query_vector=query_vector,
                top_k=fetch_k,
                filter=build_s3_filter(filter, namespace),
                return_metadata=True,
                return_distance=True,
            )
            if explanation is not None:
                explanation.timings_ms["vector_search"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )
                explanation.candidate_counts["retrieved"] = len(raw)
            if not raw:
                return []

            hits = [(self._split_key(v["key"])[1], v.get("distance")) for v in raw]
            ids = [h[0] for h in hits]
            stage_t0 = time.perf_counter()
            hydrated = self._docs.get_many(namespace, ids)

            if explanation is not None:
                explanation.timings_ms["hydration"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )
                explanation.candidate_counts["hydrated"] = len(hydrated)

            vec_by_key = {}
            if needs_vectors:
                stage_t0 = time.perf_counter()
                vec_by_key = self._vectors.get_vectors(
                    [self._s3_key(namespace, doc_id) for doc_id in ids]
                )

                if explanation is not None:
                    explanation.timings_ms["vector_fetch"] = round(
                        (time.perf_counter() - stage_t0) * 1000, 3
                    )
                    explanation.candidate_counts["vectors_fetched"] = len(vec_by_key)

            results = []
            for doc_id, distance in hits:
                doc = hydrated.get(doc_id, {})
                vec = (
                    vec_by_key.get(self._s3_key(namespace, doc_id), {}).get("vector")
                    if vec_by_key
                    else None
                )
                results.append(
                    SearchResult(
                        id=doc_id,
                        score=distance_to_score(distance, self.config.distance_metric)
                        if distance is not None
                        else 0.0,
                        distance=distance,
                        text=doc.get("text"),
                        metadata=doc.get("metadata", {}),
                        vector=vec,
                    )
                )

        if rescore is not None:
            stage_t0 = time.perf_counter()
            results = self._apply_rescore(query_vector, results, rescore)

            if explanation is not None:
                explanation.timings_ms["rescore"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )
                explanation.candidate_counts["rescored"] = len(results)
        if rerank == "mmr":
            stage_t0 = time.perf_counter()
            results = maximal_marginal_relevance(
                results,
                query_vector,
                top_k=top_k,
                lambda_mult=mmr_lambda,
            )

            if explanation is not None:
                explanation.timings_ms["rerank"] = round((time.perf_counter() - stage_t0) * 1000, 3)
                explanation.candidate_counts["reranked"] = len(results)

        elif rerank == "cross-encoder":
            stage_t0 = time.perf_counter()
            results = self._cross_encoder_rerank(
                query,
                results,
                top_k=top_k,
            )

            if explanation is not None:
                explanation.timings_ms["rerank"] = round((time.perf_counter() - stage_t0) * 1000, 3)
                explanation.candidate_counts["reranked"] = len(results)

        else:
            results = results[:top_k]

        if normalize_scores and results:
            stage_t0 = time.perf_counter()
            normalized = normalize_metric_scores(np.asarray([r.score for r in results]))
            for result, normalized_score in zip(results, normalized):
                result.score = float(normalized_score)

            if explanation is not None:
                explanation.timings_ms["normalize_scores"] = round(
                    (time.perf_counter() - stage_t0) * 1000, 3
                )

        if not include_vectors:
            for r in results:
                r.vector = None

        return results

    @staticmethod
    def _rescore_label(rescore: RescoreSpec | None) -> str | None:
        if rescore is None:
            return None
        return rescore if isinstance(rescore, str) else "composite"

    def _record_search(
        self, tel, t0, namespace, top_k, results, cache_hit, filter, rescore, rerank, query
    ) -> None:
        if tel is None:
            return
        scores = [r.score for r in results] if results else []
        tel.record(
            tel.new_event(
                "search",
                namespace=namespace,
                top_k=top_k,
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                n_results=len(results),
                cache_hit=cache_hit,
                filtered=bool(filter),
                rescore=self._rescore_label(rescore),
                rerank=rerank,
                score_top=round(max(scores), 4) if scores else None,
                score_mean=round(sum(scores) / len(scores), 4) if scores else None,
                query_preview=(query[:80] if (query and tel.capture_text) else None),
            )
        )

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

    def _cross_encoder_rerank(
        self,
        query: str | None,
        results: list[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if query is None:
            raise ConfigurationError(
                "Cross-encoder reranking requires a text query. "
                "Provide 'query' instead of a raw 'vector'."
            )

        if any(result.text is None for result in results):
            raise ConfigurationError("Cross-encoder reranking requires document text.")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise MissingDependencyError(
                "Cross-encoder reranking",
                "sentence-transformers",
                "rerank",
            ) from exc

        encoder = self._cross_encoder
        if encoder is None:
            encoder = CrossEncoder(self.config.cross_encoder_model)
            self._cross_encoder = encoder

        pairs = [(query, result.text) for result in results]
        scores = encoder.predict(pairs)

        reranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for result, score in reranked:
            result.score = float(score)

        return [result for result, _ in reranked[:top_k]]

    def search_stream(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 50,
        namespace: str = "default",
        filter: Metadata | None = None,
        page_size: int | None = None,
    ) -> Iterator[SearchResult]:
        """Stream results to the agent page-by-page as S3 Vectors returns them.

        A generator: the caller (agent) can start consuming the first hits before
        the full result set is retrieved. Reranking/rescoring are not applied in
        streaming mode (they need the whole candidate set).

        Parameters
        ----------
        page_size:
            Optional client-side chunk size for DynamoDB hydration batches.
            Defaults to ``DynavecConfig.top_k_page_size``. ``None`` uses native
            Amazon S3 Vectors pages (at most 100). Yields one hit at a time
            regardless; this does not change S3 Vectors page size.
        """
        query_vector = self._resolve_query_vector(query, vector)
        yielded = 0
        effective_page_size = page_size if page_size is not None else self.config.top_k_page_size
        for page in self._vectors.query_pages(
            query_vector=query_vector,
            top_k=top_k,
            filter=build_s3_filter(filter, namespace),
            return_metadata=True,
            return_distance=True,
            page_size=effective_page_size,
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
                    if distance is not None
                    else 0.0,
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

    def as_multiquery_retriever(
        self,
        generate_queries=None,
        *,
        llm_generate_queries=None,
        namespace: str = "default",
        **kw,
    ):
        """Create a :class:`~dynavec.retrievers.MultiQueryRetriever` bound to this client."""
        from .retrievers import MultiQueryRetriever

        return MultiQueryRetriever(
            self,
            generate_queries=generate_queries,
            llm_generate_queries=llm_generate_queries,
            namespace=namespace,
            **kw,
        )

    def as_hyde_retriever(
        self,
        generate_hypothetical=None,
        *,
        llm_generate_hypothetical=None,
        namespace: str = "default",
        **kw,
    ):
        """Create a :class:`~dynavec.retrievers.HyDERetriever` bound to this client."""
        from .retrievers import HyDERetriever

        return HyDERetriever(
            self,
            generate_hypothetical=generate_hypothetical,
            llm_generate_hypothetical=llm_generate_hypothetical,
            namespace=namespace,
            **kw,
        )

    def _resolve_query_vector(self, query: str | None, vector: list[float] | None) -> list[float]:
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

    def list_vectors(
        self,
        namespace: str | None = None,
        *,
        include_vectors: bool = False,
        hydrate: bool = True,
        page_size: int | None = None,
    ) -> Iterator[SearchResult]:
        """Stream every stored vector, one at a time, for maintenance jobs.

        A generator over the whole index — bulk delete, re-embedding, auditing.
        Only a single page is ever held in memory, so this is safe over indexes
        holding millions of vectors. Do not build a list from it unless you
        already know the index is small.

        Parameters
        ----------
        namespace:
            Restrict the walk to one namespace. ``None`` (the default) walks
            every namespace in the index. Amazon S3 Vectors ``ListVectors``
            takes no server-side ``filter``, so unlike :meth:`search` the scope
            is applied client-side on the vector key, which carries the same
            namespace that :func:`build_s3_filter` matches via the
            ``_dv_ns`` metadata tag.
        include_vectors:
            Carry the raw embedding on each result (extra bandwidth).
        hydrate:
            Fetch text and full metadata from DynamoDB, one BatchGetItem per
            page. Set False for jobs that only need ids: results then carry the
            smaller *filterable* metadata subset mirrored into S3 Vectors, and
            text only where ``store_text_in_s3vectors`` mirrored it.
        page_size:
            Service-side page size (``maxResults``, 1-1000).
        """
        for page in self._vectors.list_pages(
            return_data=include_vectors,
            return_metadata=not hydrate,
            page_size=page_size,
        ):
            scoped = []
            for v in page:
                key_ns, doc_id = self._split_key(v["key"])
                if namespace is not None and key_ns != namespace:
                    continue
                scoped.append((key_ns, doc_id, v))
            if not scoped:
                continue

            hydrated: dict[str, dict[str, dict[str, Any]]] = {}
            if hydrate:
                by_ns: dict[str, list[str]] = {}
                for key_ns, doc_id, _ in scoped:
                    by_ns.setdefault(key_ns, []).append(doc_id)
                for ns, ids in by_ns.items():
                    hydrated[ns] = self._docs.get_many(ns, ids)

            for key_ns, doc_id, v in scoped:
                if hydrate:
                    doc = hydrated.get(key_ns, {}).get(doc_id, {})
                    text = doc.get("text")
                    metadata = doc.get("metadata", {})
                else:
                    metadata = dict(v.get("metadata") or {})
                    metadata.pop(NS_METADATA_KEY, None)
                    text = metadata.pop(TEXT_METADATA_KEY, None)
                yield SearchResult(
                    id=doc_id,
                    score=1.0,
                    text=text,
                    metadata=metadata,
                    vector=(v.get("data") or {}).get("float32") if include_vectors else None,
                )

    # -------------------------------------------------------------- graph / ER
    def graph_add_node(self, entity_id, *, namespace="default", ntype=None, props=None):
        """Create/update a graph entity (a 'meaning' node)."""
        self.graph.add_node(namespace, entity_id, ntype, props)

    def graph_add_edge(self, src, relation, dst, *, namespace="default", bidirectional=False):
        """Relate two entities: ``(src) -[relation]-> (dst)``."""
        self.graph.add_edge(namespace, src, relation, dst)
        if bidirectional:
            self.graph.add_edge(namespace, dst, relation, src)

    def graph_delete_node(self, entity_id, *, namespace="default"):
        """Delete an entity with its outbound and inbound edges (idempotent).

        Linked documents and their embeddings are left untouched. Returns the
        number of inbound edges removed. Finding those scans the namespace.
        """
        return self.graph.delete_node(namespace, entity_id)

    def graph_delete_edge(self, src, relation, dst, *, namespace="default", bidirectional=False):
        """Remove ``(src) -[relation]-> (dst)`` (idempotent); return edges removed."""
        removed = self.graph.delete_edge(namespace, src, relation, dst)
        if bidirectional:
            removed += self.graph.delete_edge(namespace, dst, relation, src)
        return removed

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
    def graph_shortest_path(
        self,
        src_entity_id,
        dst_entity_id,
        *,
        namespace="default",
        relation=None,
        hops=10,
    ):
        # Use BFS to find the shortest path between two graph entities
        # within the given hop limit.
        # Returns the path if the destination is reachable; otherwise returns an empty list.

        if src_entity_id == dst_entity_id:
            return [src_entity_id]

        visited = {src_entity_id}
        frontier = [src_entity_id]

        shortest_path_for_node = {
            src_entity_id: [src_entity_id]
        }

        for _ in range(hops):
            nxt = []

            for node in frontier:
                for nb in self.graph.neighbors(namespace, node, relation):
                    if nb in visited:
                        continue

                    visited.add(nb)
                    nxt.append(nb)

                    shortest_path_for_node[nb] = (
                        shortest_path_for_node[node] + [nb]
                    )

                    if nb == dst_entity_id:
                        return shortest_path_for_node[nb]

            frontier = nxt

            if not frontier:
                break

        return []
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

        vec_by_key = self._vectors.get_vectors([self._s3_key(namespace, d) for d in doc_ids])
        hydrated = self._docs.get_many(namespace, doc_ids)

        scored = [
            d for d in doc_ids if vec_by_key.get(self._s3_key(namespace, d), {}).get("vector")
        ]
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
                    id=d,
                    score=float(scores[int(i)]),
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                )
            )
        return out

    def hybrid_graph_search(
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
        weight: float = 1.0,
    ) -> list[SearchResult]:
        """Fuse plain ANN and graph-scoped search results with RRF.

        ``weight`` controls the contribution of graph search relative to
        plain ANN search. ANN always has a weight of ``1.0``.
        """
        ann_results = self.search(
            query=query,
            vector=vector,
            top_k=top_k,
            namespace=namespace,
        )

        graph_results = self.graph_search(
            query=query,
            seed_entities=seed_entities,
            vector=vector,
            namespace=namespace,
            relation=relation,
            hops=hops,
            top_k=top_k,
            metric=metric,
        )

        return reciprocal_rank_fusion(
            [ann_results, graph_results],
            weights=[1.0, weight],
        )

    def delete(self, ids: list[str], namespace: str = "default") -> None:
        """Delete documents from both stores (and the hot tier, if enabled)."""
        keys = [self._s3_key(namespace, doc_id) for doc_id in ids]
        self._run_parallel(
            [
                lambda: self._vectors.delete_vectors(keys),
                lambda: self._docs.delete_many(namespace, ids),
            ]
        )
        if self._hot is not None:
            self._hot.delete(namespace, ids)
        self._invalidate_cache(namespace)

    # ------------------------------------------------------------- hot tier
    def warm(self, namespace: str = "default") -> int:
        """Load a namespace into the in-memory hot tier and make it authoritative.

        Scans the namespace's vectors from S3 Vectors, hydrates their text from
        DynamoDB, and holds them in RAM so subsequent searches skip both stores.
        Requires ``DynavecConfig.hot_tier=True``. Returns the number of vectors
        loaded, or ``0`` if the namespace exceeds ``hot_tier_max_vectors`` (in
        which case it stays on the S3 path). Call again to reconcile after
        out-of-band writes.
        """
        if self._hot is None:
            raise ConfigurationError(
                "Hot tier is disabled. Set DynavecConfig(hot_tier=True) to use warm()."
            )
        items = [
            (doc["id"], doc["vector"], doc["text"], doc["metadata"])
            for doc in self.iter_namespace(namespace)
        ]
        return len(items) if self._hot.load(namespace, items) else 0

    def hot_stats(self) -> dict[str, Any] | None:
        """Residency stats for the hot tier, or ``None`` if it's disabled."""
        return self._hot.stats() if self._hot is not None else None

    # ------------------------------------------------------------- export / import
    def iter_namespace(self, namespace: str = "default") -> Iterator[dict[str, Any]]:
        """Yield all documents and vectors for ``namespace``.

        Each item is a dictionary with keys:
        - ``id``: str
        - ``vector``: list[float]
        - ``text``: str | None
        - ``metadata``: dict[str, Any]
        """
        for result in self.list_vectors(namespace=namespace, include_vectors=True, hydrate=True):
            yield {
                "id": result.id,
                "vector": result.vector,
                "text": result.text,
                "metadata": result.metadata,
            }

    def export_namespace(
        self,
        output: str | Path | TextIO,
        *,
        namespace: str = "default",
    ) -> int:
        """Export all vectors and documents for a namespace as JSON Lines (JSONL).

        Parameters
        ----------
        output:
            File path (str or Path) or writable text stream (e.g. sys.stdout).
        namespace:
            The namespace to dump (default: "default").

        Returns
        -------
        int
            Number of documents exported.
        """
        count = 0
        if isinstance(output, (str, Path)):
            with open(output, "w", encoding="utf-8") as f:
                for item in self.iter_namespace(namespace):
                    f.write(json.dumps(item, default=str) + "\n")
                    count += 1
        else:
            for item in self.iter_namespace(namespace):
                output.write(json.dumps(item, default=str) + "\n")
                count += 1
        return count

    def import_namespace(
        self,
        input: str | Path | TextIO,
        *,
        namespace: str = "default",
        batch_size: int = 100,
    ) -> int:
        """Import vectors and documents from a JSON Lines (JSONL) source into a namespace.

        Parameters
        ----------
        input:
            File path (str or Path) or readable text stream (e.g. sys.stdin).
        namespace:
            Target namespace to restore into (default: "default").
        batch_size:
            Batch size for upserting records (default: 100).

        Returns
        -------
        int
            Number of documents imported.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        def _process(stream: TextIO) -> int:
            count = 0
            batch: list[Document] = []
            for line_no, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

                if "id" not in obj or "vector" not in obj:
                    raise ValueError(f"Missing required 'id' or 'vector' field at line {line_no}")

                doc = Document(
                    id=str(obj["id"]),
                    vector=obj["vector"],
                    text=obj.get("text"),
                    metadata=obj.get("metadata") or {},
                )
                batch.append(doc)
                if len(batch) >= batch_size:
                    self.upsert(batch, namespace=namespace)
                    count += len(batch)
                    batch = []

            if batch:
                self.upsert(batch, namespace=namespace)
                count += len(batch)
            return count

        if isinstance(input, (str, Path)):
            with open(input, encoding="utf-8") as f:
                return _process(f)
        return _process(input)
