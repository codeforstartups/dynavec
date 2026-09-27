"""Haystack integration for Dynavec."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dynavec.exceptions import MissingDependencyError

try:
    from haystack import Document, component
    from haystack.document_stores.errors import DuplicateDocumentError
    from haystack.document_stores.types import DuplicatePolicy
except ImportError as exc:
    raise MissingDependencyError(
        "DynavecDocumentStore",
        "haystack-ai",
        "haystack",
    ) from exc

from dynavec.client import Dynavec
from dynavec.config import DynavecConfig
from dynavec.models import Document as DynavecDocument


def _convert_filter(
    filters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not filters:
        return None

    if "field" in filters and "operator" in filters and "value" in filters:
        field = filters["field"]

        if field.startswith("meta."):
            field = field[5:]

        if filters["operator"] == "==":
            return {field: filters["value"]}

    return filters
class DynavecDocumentStore:
    """Haystack DocumentStore backed by Dynavec."""
    def __init__(self, client: Dynavec, namespace: str = "default") -> None:
        self.client = client
        self.namespace = namespace

    def count_documents(self) -> int:
        """Return the number of documents stored in the namespace."""
        return sum(
            1
            for _ in self.client.list_vectors(
                namespace=self.namespace,
                hydrate=False,
            )
        )
    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """Write Haystack documents to Dynavec."""
        if not documents:
            return 0

        ids = [document.id for document in documents]
        existing = {
            result.id
            for result in self.client.get(
                ids,
                namespace=self.namespace,
            )
        }

        if policy == DuplicatePolicy.SKIP:
            documents = [document for document in documents if document.id not in existing]

        elif policy == DuplicatePolicy.FAIL:
            duplicates = [document.id for document in documents if document.id in existing]
            if duplicates:
                raise DuplicateDocumentError(
                    f"Documents with IDs already exist: {duplicates}"
                )

        dynavec_documents = [
            DynavecDocument(
                id=document.id,
                text=document.content,
                vector=document.embedding,
                metadata=document.meta,
            )
            for document in documents
        ]

        if dynavec_documents:
            self.client.upsert(
                dynavec_documents,
                namespace=self.namespace,
            )

        return len(dynavec_documents)

    def filter_documents(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return documents matching metadata filters."""
        documents = []
        filters = _convert_filter(filters)

        for result in self.client.list_vectors(
            namespace=self.namespace,
            hydrate=True,
        ):
            if filters and any(result.metadata.get(key) != value for key, value in filters.items()
            ):
                continue

            documents.append(
                Document(
                    id=result.id,
                    content=result.text,
                    meta=result.metadata,
                    embedding=result.vector,
                )
            )

        return documents

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document store configuration."""
        return {
            "type": "dynavec.integrations.haystack.DynavecDocumentStore",
            "init_parameters": {
                "config": asdict(self.client.config),
                "namespace": self.namespace,
            },
    }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynavecDocumentStore:
        """Recreate a document store from serialized configuration."""
        parameters = data["init_parameters"]

        config = DynavecConfig(**parameters["config"])
        client = Dynavec(config)

        return cls(
            client=client,
            namespace=parameters.get("namespace", "default"),
        )



class DynavecRetriever:
    """Haystack retriever backed by Dynavec."""

    def __init__(
        self,
        client: Dynavec,
        namespace: str = "default",
        top_k: int = 10,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """Retrieve documents from Dynavec using a query embedding."""
        top_k = self.top_k if top_k is None else top_k

        results = self.client.search(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=_convert_filter(filters),
        )

        documents = [
            Document(
                id=result.id,
                content=result.text,
                meta=result.metadata,
                embedding=result.vector,
            )
            for result in results
        ]

        return {"documents": documents}
