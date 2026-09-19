"""Namespace views: ergonomic per-namespace RAG handles.

``db.namespace("kb")`` returns a lightweight object whose ``upsert`` / ``search``
/ ``update`` / ``get`` / ``delete`` are all bound to that namespace, so agent
code never repeats ``namespace=`` and different tenants/collections stay cleanly
separated on the same infrastructure.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from .models import Document, SearchResult, UpsertResult

if TYPE_CHECKING:
    from .client import Dynavec


class NamespaceView:
    """A :class:`~dynavec.client.Dynavec` proxy pinned to one namespace."""

    __slots__ = ("_db", "_ns")

    def __init__(self, db: Dynavec, namespace: str) -> None:
        self._db = db
        self._ns = namespace

    @property
    def namespace(self) -> str:
        return self._ns

    def upsert(
        self,
        documents: Sequence[Document | dict[str, Any]] | None,
        **kw: Any,
    ) -> UpsertResult:
        return self._db.upsert(documents, namespace=self._ns, **kw)

    def update(self, id: str, **kw: Any) -> UpsertResult:
        return self._db.update(id, namespace=self._ns, **kw)

    def search(self, query: str | None = None, **kw: Any) -> list[SearchResult]:
        return self._db.search(query, namespace=self._ns, **kw)

    def search_stream(self, query: str | None = None, **kw: Any) -> Iterator[SearchResult]:
        yield from self._db.search_stream(query, namespace=self._ns, **kw)

    def get(self, ids: list[str], **kw: Any) -> list[SearchResult]:
        return self._db.get(ids, namespace=self._ns, **kw)

    def delete(self, ids: list[str], **kw: Any) -> None:
        self._db.delete(ids, namespace=self._ns, **kw)

    def as_multiquery_retriever(
        self, generate_queries: Any = None, *, llm_generate_queries: Any = None, **kw: Any
    ) -> Any:
        """Create a :class:`~dynavec.retrievers.MultiQueryRetriever` pinned to this namespace."""
        from .retrievers import MultiQueryRetriever

        return MultiQueryRetriever(
            self,
            generate_queries=generate_queries,
            llm_generate_queries=llm_generate_queries,
            **kw,
        )

    def as_hyde_retriever(
        self, generate_hypothetical: Any = None, *, llm_generate_hypothetical: Any = None, **kw: Any
    ) -> Any:
        """Create a :class:`~dynavec.retrievers.HyDERetriever` pinned to this namespace."""
        from .retrievers import HyDERetriever

        return HyDERetriever(
            self,
            generate_hypothetical=generate_hypothetical,
            llm_generate_hypothetical=llm_generate_hypothetical,
            **kw,
        )

    def export_namespace(self, output: Any, **kw: Any) -> int:
        return self._db.export_namespace(output, namespace=self._ns, **kw)

    def import_namespace(self, input: Any, **kw: Any) -> int:
        return self._db.import_namespace(input, namespace=self._ns, **kw)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._db.iter_namespace(namespace=self._ns)

    def __repr__(self) -> str:
        return f"NamespaceView(namespace={self._ns!r})"
