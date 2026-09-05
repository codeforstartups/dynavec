"""LlamaIndex ``VectorStore`` adapter for dynavec.

    from dynavec import Dynavec, DynavecConfig
    from dynavec.integrations.llamaindex import DynavecLlamaStore
    from llama_index.core import VectorStoreIndex, StorageContext

    store = DynavecLlamaStore(dynavec_client, namespace="kb")
    ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(docs, storage_context=ctx)

Nodes already carry their embeddings from LlamaIndex, so dynavec stores those
vectors directly (no re-embedding).
"""

from __future__ import annotations

from typing import Any, List

from ..client import Dynavec
from ..exceptions import MissingDependencyError
from ..models import Document as DVDocument

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise MissingDependencyError(
        "DynavecLlamaStore", "llama-index-core", "all"
    ) from exc


class DynavecLlamaStore(BasePydanticVectorStore):
    """Minimal LlamaIndex vector store backed by a :class:`Dynavec` client."""

    stores_text: bool = True
    flat_metadata: bool = False

    _client: Dynavec
    _namespace: str

    def __init__(self, client: Dynavec, namespace: str = "default") -> None:
        super().__init__()
        self._client = client
        self._namespace = namespace

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        docs = []
        for node in nodes:
            meta = node.metadata or {}
            meta = {**meta, "_node_content": node.get_content()}
            docs.append(
                DVDocument(
                    id=node.node_id,
                    text=node.get_content(),
                    vector=node.get_embedding(),
                    metadata=meta,
                )
            )
        self._client.upsert(docs, namespace=self._namespace)
        return [n.node_id for n in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._client.delete([ref_doc_id], namespace=self._namespace)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        flt = None
        if query.filters is not None:
            flt = {f.key: f.value for f in query.filters.filters}

        results = self._client.search(
            vector=query.query_embedding,
            top_k=query.similarity_top_k,
            namespace=self._namespace,
            filter=flt,
        )

        nodes, ids, scores = [], [], []
        for r in results:
            nodes.append(TextNode(id_=r.id, text=r.text or "", metadata=r.metadata))
            ids.append(r.id)
            scores.append(r.score)
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)
