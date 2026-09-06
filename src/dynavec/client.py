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
3. ``BatchGetItem`` in DynamoDB to hydrate full text + metadata
4. (optional) client-side rescore, MMR rerank, or streaming delivery

Concurrency
-----------
The workload is I/O-bound, so a ``ThreadPoolExecutor`` is used for batched
writes and multi-query reads.
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
from .exceptions import (
    ConfigurationError,
    DimensionMismatchError,
    NotFoundError,
)
from .graph import GraphStore
from .metadata import (
    build_s3_filter,
    generate_auto_metadata,
    split_metadata,
)
from .metrics import rescore as metric_rescore
from .metrics import score as metric_score
from .models import (
    Document,
    NamespaceStats,
    SearchResult,
    UpsertResult,
)
from .namespace import NamespaceView
from .provisioning import provision_all
from .retrieval import (
    distance_to_score,
    maximal_marginal_relevance,
)
from .stores import DynamoDBStore, S3VectorsStore
from .transforms import TransformContext, as_pipeline
from .utils import chunked


Metadata = dict[str, Any]

_KEY_SEP = "#"
_S3_PUT_CHUNK = 500
_DDB_CHUNK = 500

RescoreSpec = Union[str, dict]


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

        self._vectors = S3VectorsStore(
            config,
            boto_session=self._session,
        )

        self._docs = DynamoDBStore(
            config,
            boto_session=self._session,
        )

        self._default_transform = as_pipeline(transform)
        self._cache = cache
        self._graph_store: GraphStore | None = None
        self._pool: ThreadPoolExecutor | None = None

        if (
            embedder is not None
            and embedder.dimension != config.dimension
        ):
            raise ConfigurationError(
                f"Embedder dimension ({embedder.dimension}) "
                f"!= index dimension ({config.dimension}). "
                "Fix the embedder or DynavecConfig.dimension."
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
        """Create the worker pool lazily."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )

        return self._pool

    def close(self) -> None:
        """Close background resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "Dynavec":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- setup

    def provision(self) -> None:
        """Create S3 Vector and DynamoDB resources."""
        provision_all(
            self.config,
            boto_session=self._session,
        )

    # ---------------------------------------------------------- namespaces

    def namespace(
        self,
        namespace: str,
    ) -> NamespaceView:
        """Return a view with all operations bound to a namespace."""
        return NamespaceView(
            self,
            namespace,
        )

    def list_namespaces(
        self,
        search: str | None = None,
    ) -> list[str]:
        """List namespaces, optionally filtered by search."""
        return self._docs.list_namespaces(
            search=search
        )

    def namespace_stats(
        self,
        namespace: str,
    ) -> NamespaceStats:
        """Return statistics for a namespace."""
        stats = self._docs.namespace_stats(namespace)

        return NamespaceStats(
            namespace=namespace,
            document_count=stats["document_count"],
            text_bytes=stats["text_bytes"],
        )

    def explore_namespaces(
        self,
        search: str | None = None,
    ) -> list[NamespaceStats]:
        """Return statistics for all matching namespaces."""
        namespaces = self.list_namespaces(
            search=search
        )

        return [
            self.namespace_stats(namespace)
            for namespace in namespaces
        ]

    # --------------------------------------------------------------- graph

    @property
    def graph(self) -> GraphStore:
        """Lazily build the knowledge graph store."""
        if self._graph_store is None:
            self._graph_store = GraphStore(
                self.config,
                boto_session=self._session,
            )

        return self._graph_store

    # ---------------------------------------------------------- key helpers

    def _s3_key(
        self,
        namespace: str,
        doc_id: str,
    ) -> str:
        return f"{namespace}{_KEY_SEP}{doc_id}"

    def _split_key(
        self,
        key: str,
    ) -> tuple[str, str]:
        namespace, _, doc_id = key.partition(
            _KEY_SEP
        )

        return namespace, doc_id

    # ------------------------------------------------------------ parallel

    def _run_parallel(
        self,
        tasks: list,
    ) -> None:
        """Run zero-argument callables."""

        if not tasks:
            return

        if (
            not self.config.parallel_writes
            or len(tasks) == 1
        ):
            for task in tasks:
                task()

            return

        futures = [
            self._executor.submit(task)
            for task in tasks
        ]

        for future in futures:
            future.result()

    # ---------------------------------------------------------- write path

    def _prepare(
        self,
        docs: list[Document],
        namespace: str,
        auto_metadata: bool,
        transform,
    ) -> tuple[list[tuple], list[tuple], list[str]]:

        pipeline = (
            as_pipeline(transform)
            or self._default_transform
        )

        # 1. Apply transforms
        if pipeline is not None:
            for document in docs:

                ctx = pipeline(
                    TransformContext(
                        id=document.id,
                        text=document.text,
                        vector=document.vector,
                        metadata=dict(document.metadata),
                        namespace=namespace,
                    )
                )

                document.text = ctx.text
                document.vector = ctx.vector
                document.metadata = ctx.metadata

        # 2. Embed documents without vectors
        to_embed = [
            (index, document.text)
            for index, document in enumerate(docs)
            if document.vector is None
        ]

        if to_embed:

            if self.embedder is None:
                raise ConfigurationError(
                    "Some documents have no vector and no "
                    "embedder is configured. Pass an embedder "
                    "to Dynavec(...) or provide precomputed vectors."
                )

            texts = [
                text
                for _, text in to_embed
            ]

            if any(
                text is None
                for text in texts
            ):
                raise ConfigurationError(
                    "A document has neither text nor vector."
                )

            vectors = self.embedder.embed_documents(
                texts
            )

            for (
                (index, _),
                vector,
            ) in zip(
                to_embed,
                vectors,
            ):
                docs[index].vector = vector

        # 3. Validate and build payloads

        s3_payload = []
        ddb_payload = []
        ids = []

        for document in docs:

            if document.vector is None:
                raise ConfigurationError(
                    f"Document {document.id!r} has no vector."
                )

            if (
                len(document.vector)
                != self.config.dimension
            ):
                raise DimensionMismatchError(
                    f"Document {document.id!r} vector has dimension "
                    f"{len(document.vector)}, expected "
                    f"{self.config.dimension}."
                )

            metadata = dict(
                document.metadata
            )

            if auto_metadata:
                auto = generate_auto_metadata(
                    document.text
                )

                auto.update(metadata)

                metadata = auto

            s3_meta, ddb_meta = split_metadata(
                metadata,
                self.config,
                namespace,
                document.text,
            )

            s3_payload.append(
                (
                    self._s3_key(
                        namespace,
                        document.id,
                    ),
                    document.vector,
                    s3_meta,
                )
            )

            ddb_payload.append(
                (
                    document.id,
                    document.text,
                    ddb_meta,
                )
            )

            ids.append(
                document.id
            )

        return (
            s3_payload,
            ddb_payload,
            ids,
        )

    def _write(
        self,
        namespace: str,
        s3_payload: list,
        ddb_payload: list,
    ) -> None:

        tasks = []

        for batch in chunked(
            s3_payload,
            _S3_PUT_CHUNK,
        ):
            tasks.append(
                lambda current=batch:
                self._vectors.put_vectors(
                    current
                )
            )

        for batch in chunked(
            ddb_payload,
            _DDB_CHUNK,
        ):
            tasks.append(
                lambda current=batch:
                self._docs.put_many(
                    namespace,
                    current,
                )
            )

        self._run_parallel(
            tasks
        )

    def upsert(
        self,
        documents: list[
            Document | dict
        ] | None = None,
        *,
        namespace: str = "default",
        auto_metadata: bool = False,
        transform=None,
    ) -> UpsertResult:
        """Insert or overwrite documents."""

        if not documents:
            return UpsertResult(
                count=0,
                ids=[],
            )

        docs = [
            document
            if isinstance(
                document,
                Document,
            )
            else Document(
                **document
            )
            for document in documents
        ]

        (
            s3_payload,
            ddb_payload,
            ids,
        ) = self._prepare(
            docs,
            namespace,
            auto_metadata,
            transform,
        )

        self._write(
            namespace,
            s3_payload,
            ddb_payload,
        )

        return UpsertResult(
            count=len(ids),
            ids=ids,
        )

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

        existing = self._docs.get_many(
            namespace,
            [id],
        ).get(id)

        if (
            existing is None
            and not upsert_if_missing
        ):
            raise NotFoundError(
                f"Document {id!r} not found in "
                f"namespace {namespace!r}."
            )

        existing = existing or {
            "text": None,
            "metadata": {},
        }

        new_text = (
            text
            if text is not None
            else existing.get("text")
        )

        new_vector = vector

        if new_vector is None:

            if (
                text is not None
                and self.embedder is not None
            ):
                new_vector = (
                    self.embedder.embed_documents(
                        [new_text]
                    )[0]
                )

            else:

                key = self._s3_key(
                    namespace,
                    id,
                )

                fetched = (
                    self._vectors.get_vectors(
                        [key]
                    )
                )

                got = fetched.get(
                    key
                )

                if (
                    got
                    and got.get("vector")
                    is not None
                ):
                    new_vector = got[
                        "vector"
                    ]

        if new_vector is None:
            raise ConfigurationError(
                f"Cannot resolve a vector for update "
                f"of {id!r}: pass 'vector', or 'text' "
                "with an embedder configured."
            )

        if metadata is None:
            new_metadata = existing.get(
                "metadata",
                {},
            )

        elif merge_metadata:

            new_metadata = {
                **existing.get(
                    "metadata",
                    {},
                ),
                **metadata,
            }

        else:
            new_metadata = metadata

        document = Document(
            id=id,
            text=new_text,
            vector=new_vector,
            metadata=new_metadata,
        )

        (
            s3_payload,
            ddb_payload,
            ids,
        ) = self._prepare(
            [document],
            namespace,
            auto_metadata=False,
            transform=transform,
        )

        self._write(
            namespace,
            s3_payload,
            ddb_payload,
        )

        return UpsertResult(
            count=1,
            ids=ids,
        )

    # ----------------------------------------------------------- read path

    def search(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 10,
        namespace: str = "default",
        filter: Metadata | None = None,
        rescore: RescoreSpec | None = None,
        rerank: str | None = None,
        mmr_lambda: float = 0.5,
        include_vectors: bool = False,
        use_cache: bool | None = None,
    ) -> list[SearchResult]:

        query_vector = self._resolve_query_vector(
            query,
            vector,
        )

        cache_on = (
            self._cache is not None
            if use_cache is None
            else use_cache
        )

        cache_filter = None

        if (
            cache_on
            and self._cache is not None
        ):

            cache_filter = {
                **(filter or {}),
                "__rank": {
                    "rescore": rescore,
                    "rerank": rerank,
                    "mmr": mmr_lambda,
                },
            }

            cached = self._cache.get(
                namespace,
                query_vector,
                top_k,
                cache_filter,
            )

            if cached is not None:
                return cached

        needs_vectors = (
            rerank == "mmr"
            or rescore is not None
            or include_vectors
        )

        fetch_k = (
            top_k * self.config.over_fetch
            if (
                rerank
                or rescore
            )
            else top_k
        )

        raw = self._vectors.query(
            query_vector=query_vector,
            top_k=fetch_k,
            filter=build_s3_filter(
                filter,
                namespace,
            ),
            return_metadata=True,
            return_distance=True,
        )

        if not raw:
            return []

        hits = [
            (
                self._split_key(
                    value["key"]
                )[1],
                value.get(
                    "distance"
                ),
            )
            for value in raw
        ]

        ids = [
            hit[0]
            for hit in hits
        ]

        hydrated = self._docs.get_many(
            namespace,
            ids,
        )

        vec_by_key = {}

        if needs_vectors:

            vec_by_key = (
                self._vectors.get_vectors(
                    [
                        self._s3_key(
                            namespace,
                            doc_id,
                        )
                        for doc_id in ids
                    ]
                )
            )

        results: list[
            SearchResult
        ] = []

        for (
            doc_id,
            distance,
        ) in hits:

            document = hydrated.get(
                doc_id,
                {},
            )

            vector_value = None

            if vec_by_key:

                vector_value = (
                    vec_by_key.get(
                        self._s3_key(
                            namespace,
                            doc_id,
                        ),
                        {},
                    ).get(
                        "vector"
                    )
                )

            results.append(
                SearchResult(
                    id=doc_id,
                    score=(
                        distance_to_score(
                            distance,
                            self.config.distance_metric,
                        )
                        if distance is not None
                        else 0.0
                    ),
                    distance=distance,
                    text=document.get(
                        "text"
                    ),
                    metadata=document.get(
                        "metadata",
                        {},
                    ),
                    vector=vector_value,
                )
            )

        if rescore is not None:
            results = self._apply_rescore(
                query_vector,
                results,
                rescore,
            )

        if rerank == "mmr":

            results = (
                maximal_marginal_relevance(
                    query_vector,
                    results,
                    top_k,
                    mmr_lambda,
                )
            )

        else:
            results = results[
                :top_k
            ]

        if not include_vectors:

            for result in results:
                result.vector = None

        if (
            cache_on
            and self._cache is not None
            and results
        ):

            self._cache.put(
                namespace,
                query_vector,
                top_k,
                cache_filter,
                results,
            )

        return results

    def _apply_rescore(
        self,
        query_vector: list[float],
        results: list[SearchResult],
        spec: RescoreSpec,
    ) -> list[SearchResult]:

        scored = [
            result
            for result in results
            if result.vector is not None
        ]

        if not scored:
            return results

        matrix = np.asarray(
            [
                result.vector
                for result in scored
            ],
            dtype=np.float32,
        )

        order, scores = metric_rescore(
            np.asarray(
                query_vector,
                dtype=np.float32,
            ),
            matrix,
            spec,
        )

        output = []

        for rank_pos in order:

            result = scored[
                int(rank_pos)
            ]

            result.score = float(
                scores[
                    int(rank_pos)
                ]
            )

            output.append(
                result
            )

        return output

    def search_stream(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 50,
        namespace: str = "default",
        filter: Metadata | None = None,
    ) -> Iterator[SearchResult]:

        query_vector = self._resolve_query_vector(
            query,
            vector,
        )

        yielded = 0

        for page in self._vectors.query_pages(

            query_vector=query_vector,
            top_k=top_k,

            filter=build_s3_filter(
                filter,
                namespace,
            ),

            return_metadata=True,
            return_distance=True,
        ):

            page_hits = [
                (
                    self._split_key(
                        value["key"]
                    )[1],
                    value.get(
                        "distance"
                    ),
                )
                for value in page
            ]

            hydrated = (
                self._docs.get_many(
                    namespace,
                    [
                        hit[0]
                        for hit in page_hits
                    ],
                )
            )

            for (
                doc_id,
                distance,
            ) in page_hits:

                if yielded >= top_k:
                    return

                document = hydrated.get(
                    doc_id,
                    {},
                )

                yield SearchResult(
                    id=doc_id,

                    score=(
                        distance_to_score(
                            distance,
                            self.config.distance_metric,
                        )
                        if distance is not None
                        else 0.0
                    ),

                    distance=distance,

                    text=document.get(
                        "text"
                    ),

                    metadata=document.get(
                        "metadata",
                        {},
                    ),
                )

                yielded += 1

    def search_many(
        self,
        queries: list[str],
        *,
        top_k: int = 10,
        namespace: str = "default",
        **kw,
    ) -> list[list[SearchResult]]:

        futures = [
            self._executor.submit(
                self.search,
                query,
                top_k=top_k,
                namespace=namespace,
                **kw,
            )
            for query in queries
        ]

        return [
            future.result()
            for future in futures
        ]

    def _resolve_query_vector(
        self,
        query: str | None,
        vector: list[float] | None,
    ) -> list[float]:

        if vector is not None:

            if (
                len(vector)
                != self.config.dimension
            ):
                raise DimensionMismatchError(
                    f"Query vector dimension "
                    f"{len(vector)} != "
                    f"{self.config.dimension}."
                )

            return vector

        if query is None:
            raise ValueError(
                "Provide either 'query' text "
                "or a 'vector'."
            )

        if self.embedder is None:
            raise ConfigurationError(
                "Text query requires an embedder. "
                "Pass one to Dynavec(...) or query "
                "with a precomputed 'vector'."
            )

        return self.embedder.embed_query(
            query
        )

    # --------------------------------------------------------------- CRUD

    def get(
        self,
        ids: list[str],
        namespace: str = "default",
    ) -> list[SearchResult]:

        hydrated = self._docs.get_many(
            namespace,
            ids,
        )

        return [
            SearchResult(
                id=doc_id,
                score=1.0,
                text=hydrated[
                    doc_id
                ].get(
                    "text"
                ),
                metadata=hydrated[
                    doc_id
                ].get(
                    "metadata",
                    {},
                ),
            )

            for doc_id in ids

            if doc_id in hydrated
        ]

    # ---------------------------------------------------------- graph / ER

    def graph_add_node(
        self,
        entity_id,
        *,
        namespace="default",
        ntype=None,
        props=None,
    ):
        """Create or update a graph entity."""

        self.graph.add_node(
            namespace,
            entity_id,
            ntype,
            props,
        )

    def graph_add_edge(
        self,
        src,
        relation,
        dst,
        *,
        namespace="default",
        bidirectional=False,
    ):
        """Create a relationship between graph entities."""

        self.graph.add_edge(
            namespace,
            src,
            relation,
            dst,
        )

        if bidirectional:

            self.graph.add_edge(
                namespace,
                dst,
                relation,
                src,
            )

    def graph_link(
        self,
        entity_id,
        doc_ids,
        *,
        namespace="default",
    ):
        """Attach documents to an entity."""

        self.graph.link_docs(
            namespace,
            entity_id,
            list(doc_ids),
        )

    def graph_neighbors(
        self,
        entity_id,
        *,
        namespace="default",
        relation=None,
        hops=1,
    ):

        visited = {
            entity_id
        }

        frontier = [
            entity_id
        ]

        for _ in range(hops):

            next_nodes = []

            for node in frontier:

                for neighbor in self.graph.neighbors(
                    namespace,
                    node,
                    relation,
                ):

                    if neighbor not in visited:

                        visited.add(
                            neighbor
                        )

                        next_nodes.append(
                            neighbor
                        )

            frontier = next_nodes

            if not frontier:
                break

        return [
            entity
            for entity in visited
            if entity != entity_id
        ]

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

        query_vector = (
            self._resolve_query_vector(
                query,
                vector,
            )
        )

        entities = list(
            dict.fromkeys(
                seed_entities
            )
        )

        for seed in seed_entities:

            entities += (
                self.graph_neighbors(
                    seed,
                    namespace=namespace,
                    relation=relation,
                    hops=hops,
                )
            )

        entities = list(
            dict.fromkeys(
                entities
            )
        )

        doc_ids = self.graph.get_docs(
            namespace,
            entities,
        )

        if not doc_ids:
            return []

        vec_by_key = (
            self._vectors.get_vectors(
                [
                    self._s3_key(
                        namespace,
                        doc_id,
                    )
                    for doc_id in doc_ids
                ]
            )
        )

        hydrated = (
            self._docs.get_many(
                namespace,
                doc_ids,
            )
        )

        scored = [
            doc_id
            for doc_id in doc_ids
            if vec_by_key.get(
                self._s3_key(
                    namespace,
                    doc_id,
                ),
                {},
            ).get(
                "vector"
            )
        ]

        if not scored:
            return []

        matrix = np.asarray(
            [
                vec_by_key[
                    self._s3_key(
                        namespace,
                        doc_id,
                    )
                ]["vector"]

                for doc_id in scored
            ],
            dtype=np.float32,
        )

        scores = metric_score(
            np.asarray(
                query_vector,
                dtype=np.float32,
            ),
            matrix,
            metric,
        )

        order = np.argsort(
            -scores
        )[:top_k]

        output = []

        for index in order:

            doc_id = scored[
                int(index)
            ]

            document = hydrated.get(
                doc_id,
                {},
            )

            output.append(
                SearchResult(
                    id=doc_id,

                    score=float(
                        scores[
                            int(index)
                        ]
                    ),

                    text=document.get(
                        "text"
                    ),

                    metadata=document.get(
                        "metadata",
                        {},
                    ),
                )
            )

        return output

    def delete(
        self,
        ids: list[str],
        namespace: str = "default",
    ) -> None:
        """Delete documents from both stores."""

        keys = [
            self._s3_key(
                namespace,
                doc_id,
            )
            for doc_id in ids
        ]

        self._run_parallel(
            [
                lambda: self._vectors.delete_vectors(
                    keys
                ),

                lambda: self._docs.delete_many(
                    namespace,
                    ids,
                ),
            ]
        )