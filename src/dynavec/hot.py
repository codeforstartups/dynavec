"""In-memory hot tier — Pinecone-class latency without a paid cluster.

S3 Vectors is cost-optimized, object-backed ANN: cheap and serverless, but its
per-query server time is hundreds of milliseconds. For the *hot* working set,
dynavec can keep vectors in RAM (via :class:`~dynavec.spfresh.SPFreshHotIndex`)
so a warmed namespace is served **entirely from memory** — no S3 Vectors query
and no DynamoDB hydration, since the hot index already holds text + metadata.

Correctness first: a namespace is served from the hot tier **only** when it is
*authoritative* — every one of its vectors is resident in RAM. A namespace
becomes authoritative after :meth:`Dynavec.warm` loads it from S3 Vectors within
the RAM budget; write-through keeps it current. Any namespace that is not
authoritative (never warmed, or grown past the RAM cap) transparently falls back
to the S3 Vectors path — so the hot tier can only ever make queries faster, never
wrong.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import DynavecConfig
from .models import SearchResult
from .spfresh import SPFreshConfig, SPFreshHotIndex

# --- MongoDB-style metadata matcher (mirrors the S3 Vectors filter dialect) ---

_COMPARATORS = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
    "$in": lambda a, b: a in b,
    "$nin": lambda a, b: a not in b,
}


class UnsupportedFilter(Exception):
    """Raised when a filter uses an operator the in-memory matcher can't honor.

    The client catches this and falls back to the S3 Vectors path so results
    stay correct.
    """


def _match_field(value: Any, condition: Any) -> bool:
    """Match one metadata field ``value`` against a user ``condition``."""
    if isinstance(condition, dict):
        for op, operand in condition.items():
            cmp = _COMPARATORS.get(op)
            if cmp is None:
                raise UnsupportedFilter(op)
            if not cmp(value, operand):
                return False
        return True
    return value == condition  # bare {k: v} means equality


def matches(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    """Return True if ``metadata`` satisfies the MongoDB-style ``flt``.

    Raises :class:`UnsupportedFilter` for operators outside the supported set so
    the caller can fall back to S3 rather than return a wrong result.
    """
    if not flt:
        return True
    for key, condition in flt.items():
        if key == "$and":
            if not all(matches(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(metadata, sub) for sub in condition):
                return False
        elif key in ("$not",):
            raise UnsupportedFilter(key)
        else:
            if not _match_field(metadata.get(key), condition):
                return False
    return True


class HotTier:
    """Per-namespace in-memory ANN cache in front of S3 Vectors.

    Thread-safe. Only *authoritative* namespaces (fully resident within the RAM
    budget) are served from memory; everything else returns ``None`` from
    :meth:`search` so the client falls back to S3 Vectors.
    """

    def __init__(self, config: DynavecConfig) -> None:
        self._config = config
        self._metric = config.distance_metric
        self._max_vectors = config.hot_tier_max_vectors
        self._n_probe = config.hot_tier_n_probe
        self._lock = threading.RLock()
        self._indexes: dict[str, SPFreshHotIndex] = {}
        self._authoritative: set[str] = set()

    # ------------------------------------------------------------------ internal
    def _new_index(self) -> SPFreshHotIndex:
        return SPFreshHotIndex(
            config=SPFreshConfig(
                dimension=self._config.dimension,
                metric=self._metric,
                n_probe=self._n_probe,
            )
        )

    @property
    def _total(self) -> int:
        return sum(len(ix) for ix in self._indexes.values())

    # ------------------------------------------------------------------ capacity
    def _demote(self, namespace: str) -> None:
        """Drop a namespace from the hot tier (frees RAM; falls back to S3)."""
        self._indexes.pop(namespace, None)
        self._authoritative.discard(namespace)

    # -------------------------------------------------------------------- writes
    def insert_many(
        self,
        namespace: str,
        items: list[tuple[str, list[float], str | None, dict[str, Any]]],
    ) -> None:
        """Write vectors through to the hot index for ``namespace``.

        Only touches namespaces that already have a hot index (i.e. were warmed
        or are being built). If the write would push total residency past the RAM
        cap, the namespace is demoted (dropped) so we never hold a partial set
        while claiming authority over it.
        """
        if not items:
            return
        with self._lock:
            index = self._indexes.get(namespace)
            if index is None:
                return  # namespace not tracked; nothing to keep in sync
            projected = self._total + len(items)
            if projected > self._max_vectors:
                self._demote(namespace)
                return
            for doc_id, vector, text, metadata in items:
                index.insert(id=doc_id, vector=vector, metadata=metadata or {}, text=text)

    def delete(self, namespace: str, ids: list[str]) -> None:
        with self._lock:
            index = self._indexes.get(namespace)
            if index is None:
                return
            for doc_id in ids:
                index.delete(doc_id)

    def load(
        self,
        namespace: str,
        items: list[tuple[str, list[float], str | None, dict[str, Any]]],
    ) -> bool:
        """Bulk-load a full namespace and mark it authoritative if it fits.

        Returns True if the namespace is now authoritative (RAM-resident), False
        if it exceeds the RAM budget (left on the S3 path).
        """
        with self._lock:
            # Free any prior residency for this namespace before recomputing budget.
            self._demote(namespace)
            if self._total + len(items) > self._max_vectors:
                return False
            index = self._new_index()
            for doc_id, vector, text, metadata in items:
                index.insert(id=doc_id, vector=vector, metadata=metadata or {}, text=text)
            self._indexes[namespace] = index
            self._authoritative.add(namespace)
            return True

    # --------------------------------------------------------------------- reads
    def is_authoritative(self, namespace: str) -> bool:
        with self._lock:
            return namespace in self._authoritative

    def search(
        self,
        namespace: str,
        query_vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult] | None:
        """Serve a search from RAM, or return ``None`` to fall back to S3.

        Returns ``None`` when the namespace is not authoritative or the filter
        uses an unsupported operator — the caller then uses the S3 path.
        """
        with self._lock:
            if namespace not in self._authoritative:
                return None
            index = self._indexes.get(namespace)
            if index is None:
                return None

        try:
            filter_fn = None
            if filter:
                filter_fn = lambda meta: matches(meta, filter)  # noqa: E731
                # Validate operators up-front so an unsupported filter falls back
                # to S3 rather than silently dropping candidates mid-scan.
                matches({}, filter)
            return index.search(
                query=query_vector,
                top_k=top_k,
                nprobe=self._n_probe,
                filter_fn=filter_fn,
            )
        except UnsupportedFilter:
            return None

    # ---------------------------------------------------------------- lifecycle
    def clear(self, namespace: str | None = None) -> None:
        with self._lock:
            if namespace is None:
                self._indexes.clear()
                self._authoritative.clear()
            else:
                self._demote(namespace)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "authoritative_namespaces": sorted(self._authoritative),
                "resident_vectors": self._total,
                "max_vectors": self._max_vectors,
                "namespaces": {ns: len(ix) for ns, ix in self._indexes.items()},
            }
