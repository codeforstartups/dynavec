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

from collections.abc import Sequence
from typing import Any

from ..client import Dynavec
from ..exceptions import MissingDependencyError
from ..models import Document as DVDocument

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise MissingDependencyError("DynavecLlamaStore", "llama-index-core", "all") from exc


_LLAMA_OPERATOR_MAP = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
}


def _translate_filter(metadata_filter: MetadataFilter) -> dict[str, Any]:
    operator = _LLAMA_OPERATOR_MAP.get(metadata_filter.operator)
    if operator is None:
        raise ValueError(
            "Unsupported LlamaIndex metadata filter operator for S3 Vectors: "
            f"{metadata_filter.operator.value}"
        )

    return {
        metadata_filter.key: {
            operator: metadata_filter.value,
        }
    }


def _translate_filters(metadata_filters: MetadataFilters) -> dict[str, Any]:
    translated = []

    for filter_ in metadata_filters.filters:
        if isinstance(filter_, MetadataFilters):
            nested = _translate_filters(filter_)
            if nested:
                translated.append(nested)
        else:
            translated.append(_translate_filter(filter_))

    if not translated:
        return {}

    condition = metadata_filters.condition or FilterCondition.AND

    if condition == FilterCondition.AND:
        return {"$and": translated}

    if condition == FilterCondition.OR:
        return {"$or": translated}

    raise ValueError(
        "Unsupported LlamaIndex metadata filter condition for S3 Vectors: "
        f"{condition.value}"
    )


class DynavecLlamaStore(BasePydanticVectorStore):
    """Minimal LlamaIndex vector store backed by a :class:`Dynavec` client."""

    stores_text: bool = True
    flat_metadata: bool = False

    _client: Dynavec
    _namespace: str

    def __init__(self, client: Dynavec, namespace: str = "default") -> None:
        super().__init__(stores_text=True)
        self._client = client
        self._namespace = namespace

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
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
            flt = _translate_filters(query.filters)

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
