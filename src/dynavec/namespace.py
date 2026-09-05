"""Namespace views: ergonomic per-namespace RAG handles.

``db.namespace("kb")`` returns a lightweight object whose ``upsert`` / ``search``
/ ``update`` / ``get`` / ``delete`` are all bound to that namespace, so agent
code never repeats ``namespace=`` and different tenants/collections stay cleanly
separated on the same infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .models import SearchResult

if TYPE_CHECKING:
    from .client import Dynavec


class NamespaceView:
    """A :class:`~dynavec.client.Dynavec` proxy pinned to one namespace."""

    __slots__ = ("_db", "_ns")

    def __init__(self, db: "Dynavec", namespace: str) -> None:
        self._db = db
        self._ns = namespace

    @property
    def namespace(self) -> str:
        return self._ns

    def upsert(self, documents, **kw) -> Any:
        return self._db.upsert(documents, namespace=self._ns, **kw)

    def update(self, *args, **kw) -> Any:
        return self._db.update(*args, namespace=self._ns, **kw)

    def search(self, query: Optional[str] = None, **kw) -> list[SearchResult]:
        return self._db.search(query, namespace=self._ns, **kw)

    def search_stream(self, query: Optional[str] = None, **kw):
        yield from self._db.search_stream(query, namespace=self._ns, **kw)

    def get(self, ids, **kw) -> list[SearchResult]:
        return self._db.get(ids, namespace=self._ns, **kw)

    def delete(self, ids, **kw) -> None:
        return self._db.delete(ids, namespace=self._ns, **kw)

    def __repr__(self) -> str:
        return f"NamespaceView(namespace={self._ns!r})"
