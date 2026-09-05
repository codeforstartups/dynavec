"""LangChain ``VectorStore`` adapter for dynavec.

    from dynavec import Dynavec, DynavecConfig
    from dynavec.integrations.langchain import DynavecVectorStore

    store = DynavecVectorStore(dynavec_client, namespace="kb")
    retriever = store.as_retriever(search_kwargs={"k": 4})

Works with any dynavec embedder, or with a LangChain ``Embeddings`` object (which
is wrapped to satisfy dynavec's :class:`~dynavec.embeddings.base.Embedder`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from ..client import Dynavec
from ..embeddings.base import Embedder
from ..exceptions import MissingDependencyError
from ..models import Document as DVDocument

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - import guard
    raise MissingDependencyError("DynavecVectorStore", "langchain-core", "langchain") from exc


class _LCEmbeddingsAdapter(Embedder):
    """Wrap a LangChain ``Embeddings`` object as a dynavec ``Embedder``."""

    def __init__(self, lc_embeddings, dimension: int) -> None:
        self._lc = lc_embeddings
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._lc.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._lc.embed_query(text)


class DynavecVectorStore(VectorStore):
    """A thin LangChain VectorStore backed by a :class:`Dynavec` client."""

    def __init__(self, client: Dynavec, namespace: str = "default") -> None:
        self._client = client
        self._namespace = namespace

    @property
    def embeddings(self):  # LangChain introspects this
        return self._client.embedder

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        docs = [
            DVDocument(id=i, text=t, metadata=m)
            for i, t, m in zip(ids, texts, metadatas)
        ]
        self._client.upsert(docs, namespace=self._namespace)
        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            return False
        self._client.delete(ids, namespace=self._namespace)
        return True

    def similarity_search(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[LCDocument]:
        results = self._client.search(
            query, top_k=k, namespace=self._namespace, filter=filter
        )
        return [
            LCDocument(page_content=r.text or "", metadata={**r.metadata, "id": r.id, "score": r.score})
            for r in results
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[tuple[LCDocument, float]]:
        results = self._client.search(
            query, top_k=k, namespace=self._namespace, filter=filter
        )
        return [
            (
                LCDocument(page_content=r.text or "", metadata={**r.metadata, "id": r.id}),
                r.score,
            )
            for r in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[LCDocument]:
        results = self._client.search(
            query,
            top_k=k,
            namespace=self._namespace,
            filter=filter,
            rerank="mmr",
            mmr_lambda=lambda_mult,
        )
        return [
            LCDocument(page_content=r.text or "", metadata={**r.metadata, "id": r.id, "score": r.score})
            for r in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding,
        metadatas: list[dict] | None = None,
        *,
        client: Dynavec | None = None,
        namespace: str = "default",
        **kwargs: Any,
    ) -> DynavecVectorStore:
        if client is None:
            raise ValueError(
                "DynavecVectorStore.from_texts requires a configured `client=Dynavec(...)`."
            )
        store = cls(client, namespace=namespace)
        store.add_texts(texts, metadatas=metadatas)
        return store
