"""Async client for Dynavec."""

from __future__ import annotations

import asyncio
from typing import Any

import aioboto3

from .config import DynavecConfig
from .embeddings.base import Embedder
from .exceptions import ConfigurationError
from .metadata import build_s3_filter, generate_auto_metadata, split_metadata
from .models import Document, SearchResult, UpsertResult
from .retrieval import distance_to_score
from .stores.async_dynamodb import AsyncDynamoDBStore
from .stores.async_s3vectors import AsyncS3VectorsStore
from .utils import KEY_SEPARATOR, decode_key_component, encode_key_component

Metadata = dict[str, Any]

class AsyncDynavec:
    """Async variant of Dynavec using aioboto3."""

    def __init__(
        self,
        config: DynavecConfig,
        embedder: Embedder | None = None,
        *,
        aioboto_session=None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._session = aioboto_session or aioboto3.Session()
        
        self._vectors = AsyncS3VectorsStore(config, aioboto_session=self._session)
        self._docs = AsyncDynamoDBStore(config, aioboto_session=self._session)

        if embedder is not None and embedder.dimension != config.dimension:
            raise ConfigurationError(
                f"Embedder dimension ({embedder.dimension}) != index dimension "
                f"({config.dimension})."
            )

    def _s3_key(self, namespace: str, doc_id: str) -> str:
        return f"{encode_key_component(namespace)}{KEY_SEPARATOR}{encode_key_component(doc_id)}"

    def _split_key(self, key: str) -> tuple[str, str]:
        namespace, _, doc_id = key.partition(KEY_SEPARATOR)
        return decode_key_component(namespace), decode_key_component(doc_id)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Run blocking embedder in a thread
        if self.embedder is None:
            raise ConfigurationError("No embedder configured.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embedder.embed_documents, texts)

    async def _resolve_query_vector(
        self, query: str | None, vector: list[float] | None
    ) -> list[float]:
        if vector is not None:
            return vector
        if query is not None:
            if self.embedder is None:
                raise ConfigurationError("Pass 'vector' or provide an embedder to use 'query'.")
            vecs = await self._embed_documents([query])
            return vecs[0]
        raise ValueError("Must provide either 'query' or 'vector'.")

    async def search(
        self,
        query: str | None = None,
        *,
        vector: list[float] | None = None,
        top_k: int = 10,
        namespace: str = "default",
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        query_vector = await self._resolve_query_vector(query, vector)

        raw = await self._vectors.query(
            query_vector=query_vector,
            top_k=top_k,
            filter=build_s3_filter(filter, namespace),
            return_metadata=True,
            return_distance=True,
        )

        if not raw:
            return []

        hits = [(self._split_key(v["key"])[1], v.get("distance")) for v in raw]
        ids = [h[0] for h in hits]
        
        hydrated = await self._docs.get_many(namespace, ids)

        results = []
        for doc_id, distance in hits:
            doc = hydrated.get(doc_id, {})
            results.append(
                SearchResult(
                    id=doc_id,
                    score=distance_to_score(distance, self.config.distance_metric)
                    if distance is not None
                    else 0.0,
                    distance=distance,
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                )
            )

        return results

    async def upsert(
        self,
        documents: list[Document | dict] | None = None,
        *,
        namespace: str = "default",
        auto_metadata: bool = False,
    ) -> UpsertResult:
        if not documents:
            return UpsertResult(count=0, ids=[])
            
        docs = [d if isinstance(d, Document) else Document(**d) for d in documents]

        to_embed = [(i, d.text) for i, d in enumerate(docs) if d.vector is None]
        if to_embed:
            texts = [t for _, t in to_embed]
            vectors = await self._embed_documents(texts)
            for (idx, _), vec in zip(to_embed, vectors):
                docs[idx].vector = vec

        s3_payload, ddb_payload, ids = [], [], []
        for d in docs:
            meta = dict(d.metadata)
            if auto_metadata:
                auto = generate_auto_metadata(d.text)
                auto.update(meta)
                meta = auto
            s3_meta, ddb_meta = split_metadata(meta, self.config, namespace, d.text)
            
            s3_payload.append((self._s3_key(namespace, d.id), d.vector, s3_meta))
            ddb_payload.append((d.id, d.text, ddb_meta))
            ids.append(d.id)

        # Run writes concurrently
        await asyncio.gather(
            self._vectors.put_vectors(s3_payload),
            self._docs.put_many(namespace, ddb_payload)
        )

        return UpsertResult(count=len(ids), ids=ids)
