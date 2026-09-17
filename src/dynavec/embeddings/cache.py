"""Pluggable cache interface for embedding vectors.

Cache key = sha256(text) + ":" + model_name
Same design as the query cache (dynavec/cache.py) — get/put → get/set,
hits/misses counters, stats() — so the codebase stays consistent.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

from .base import Vector


def _make_key(text: str, model: str) -> str:
    """sha256(text):model  — fixed-length, collision-resistant cache key."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest}:{model}"


class EmbeddingCache(ABC):
    """Abstract cache backend for embedding vectors.

    Implement ``get`` and ``set``; everything else is inherited.

    Example custom backend::

        class RedisEmbeddingCache(EmbeddingCache):
            def get(self, key):
                raw = self._r.get(key)
                return json.loads(raw) if raw else None
            def set(self, key, vector):
                self._r.set(key, json.dumps(vector))
    """

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0

    @abstractmethod
    def get(self, key: str) -> Vector | None:
        """Return cached vector or ``None`` on miss."""

    @abstractmethod
    def set(self, key: str, vector: Vector) -> None:
        """Store vector under key."""

    # --- convenience helpers (free for all subclasses) ---

    def get_for(self, text: str, model: str) -> Vector | None:
        result = self.get(_make_key(text, model))
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def set_for(self, text: str, model: str, vector: Vector) -> None:
        self.set(_make_key(text, model), vector)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


class InMemoryCache(EmbeddingCache):
    """LRU dict-backed cache — zero dependencies, lives in RAM.

    Parameters
    ----------
    max_size:
        Max number of entries. Oldest (LRU) entry is evicted when full.
        ``None`` = unlimited.
    """

    def __init__(self, max_size: int | None = 10_000) -> None:
        super().__init__()
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be non-negative")
        self.max_size = max_size
        self._store: OrderedDict[str, Vector] = OrderedDict()

    def get(self, key: str) -> Vector | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)   # mark as recently used
        return self._store[key]

    def set(self, key: str, vector: Vector) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if self.max_size is not None and len(self._store) >= self.max_size:
                self._store.popitem(last=False)   # evict LRU
            self._store[key] = vector

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()
