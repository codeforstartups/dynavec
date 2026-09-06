"""Query caching so repeated / similar searches skip the vector DB.

Three backends, pick per your infra:

* :class:`SemanticCache`  — in-process LRU that returns a cached answer when a
  new query is **cosine-similar enough** to a recent one. Zero infra, per-process.
* :class:`DynamoDBCache`  — exact-match cache in your DynamoDB table with native
  **TTL** expiry. No extra service ("if we can do it with DynamoDB, good").
* :class:`RedisCache`     — shared, sub-millisecond cache on Redis / **AWS
  ElastiCache**, great across many workers/hosts.

All expose the same ``get`` / ``put`` interface, keyed on
(namespace, query vector, top_k, filter).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from .exceptions import MissingDependencyError
from .models import SearchResult


def _signature(namespace: str, top_k: int, filter: dict | None) -> str:
    payload = json.dumps(
        {"ns": namespace, "k": top_k, "f": filter or {}}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _vec_key(query_vector, ndigits: int = 3) -> str:
    rounded = [round(float(x), ndigits) for x in query_vector]
    return hashlib.sha256(json.dumps(rounded).encode()).hexdigest()[:24]


def _serialize(results: list[SearchResult]) -> str:
    return json.dumps([r.to_dict() for r in results])


def _deserialize(blob: str) -> list[SearchResult]:
    return [
        SearchResult(
            id=d["id"], score=d["score"], distance=d.get("distance"),
            text=d.get("text"), metadata=d.get("metadata", {}),
        )
        for d in json.loads(blob)
    ]


def _deep_size(value, seen: set[int] | None = None) -> int:
    """Estimate an object graph's resident size without requiring serialization."""
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return size + _deep_size(vars(value), seen)
    return size


class BaseCache(ABC):
    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0

    @abstractmethod
    def get(self, namespace, query_vector, top_k, filter) -> list[SearchResult] | None:
        ...

    @abstractmethod
    def put(self, namespace, query_vector, top_k, filter, results) -> None:
        ...

    def stats(self) -> dict[str, int | float]:
        """Return cache hit and miss statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def reset_stats(self) -> None:
        """Reset hit and miss counters."""
        self.hits = 0
        self.misses = 0


class SemanticCache(BaseCache):
    """In-process cache that also serves *near-duplicate* queries.

    A new query reuses a cached result when its cosine similarity to a cached
    query (with the same namespace/top_k/filter) is >= ``threshold``. This trades
    a little exactness for a large latency win on paraphrases and repeats.
    """

    def __init__(
        self,
        threshold: float = 0.97,
        max_size: int = 2048,
        max_bytes: int | None = None,
    ) -> None:
        super().__init__()
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.threshold = threshold
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._size_bytes = 0
        self._lru: OrderedDict[tuple[str, str], None] = OrderedDict()
        # key: signature -> OrderedDict[vec_key -> (unit_vec, results, size_bytes)]
        self._buckets: dict[str, OrderedDict[str, tuple]] = {}

    @property
    def size_bytes(self) -> int:
        """Approximate bytes used by cached vectors and result object graphs."""
        return self._size_bytes

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def get(self, namespace, query_vector, top_k, filter):
        sig = _signature(namespace, top_k, filter)
        bucket = self._buckets.get(sig)
        if not bucket:
            self.misses += 1
            return None
        q = self._unit(np.asarray(query_vector, dtype=np.float32))
        best_key, best_sim = None, -1.0
        for vk, (vec, _res, _size) in bucket.items():
            sim = float(vec @ q)
            if sim > best_sim:
                best_sim, best_key = sim, vk
        if best_key is not None and best_sim >= self.threshold:
            entry = bucket.pop(best_key)
            bucket[best_key] = entry  # move to MRU within this signature
            self._lru.move_to_end((sig, best_key))
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def put(self, namespace, query_vector, top_k, filter, results):
        sig = _signature(namespace, top_k, filter)
        bucket = self._buckets.setdefault(sig, OrderedDict())
        vk = _vec_key(query_vector)
        vector = self._unit(np.asarray(query_vector, dtype=np.float32))
        seen: set[int] = set()
        entry_size = _deep_size(vector, seen) + _deep_size(results, seen)
        previous = bucket.pop(vk, None)
        if previous is not None:
            self._size_bytes -= previous[2]
            self._lru.pop((sig, vk), None)
        bucket[vk] = (vector, results, entry_size)
        bucket.move_to_end(vk)
        self._lru[(sig, vk)] = None
        self._size_bytes += entry_size
        while len(self._lru) > self.max_size or (
            self.max_bytes is not None and self._size_bytes > self.max_bytes
        ):
            old_sig, old_vk = self._lru.popitem(last=False)[0]
            old_bucket = self._buckets[old_sig]
            _, _, old_size = old_bucket.pop(old_vk)
            self._size_bytes -= old_size
            if not old_bucket:
                del self._buckets[old_sig]


class DynamoDBCache(BaseCache):
    """Exact-match cache persisted in the dynavec table with TTL expiry.

    Enable DynamoDB TTL on the ``ttl`` attribute of your table for automatic
    eviction (items are also treated as expired client-side as a safety net).
    """

    def __init__(self, config, boto_session=None, ttl_seconds: int = 3600) -> None:
        super().__init__()
        import boto3

        session = boto_session or boto3.Session()
        self._table = session.resource("dynamodb", region_name=config.region).Table(config.table)
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _pk(namespace, query_vector, top_k, filter) -> str:
        return f"__cache__#{_signature(namespace, top_k, filter)}#{_vec_key(query_vector)}"

    def get(self, namespace, query_vector, top_k, filter):
        resp = self._table.get_item(
            Key={"pk": self._pk(namespace, query_vector, top_k, filter)}
        )
        item = resp.get("Item")
        if not item:
            self.misses += 1
            return None
        if item.get("ttl") and int(item["ttl"]) < int(time.time()):
            self.misses += 1
            return None  # expired but not yet reaped
        self.hits += 1
        return _deserialize(item["results"])

    def put(self, namespace, query_vector, top_k, filter, results):
        self._table.put_item(
            Item={
                "pk": self._pk(namespace, query_vector, top_k, filter),
                "kind": "querycache",
                "results": _serialize(results),
                "ttl": int(time.time()) + self.ttl_seconds,
            }
        )


class RedisCache(BaseCache):
    """Shared exact-match cache on Redis / AWS ElastiCache."""

    def __init__(self, url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600, client=None):
        super().__init__()
        if client is not None:
            self._r = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError("RedisCache", "redis", "redis") from exc
            self._r = redis.Redis.from_url(url)
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(namespace, query_vector, top_k, filter) -> str:
        return f"dynavec:{_signature(namespace, top_k, filter)}:{_vec_key(query_vector)}"

    def get(self, namespace, query_vector, top_k, filter):
        blob = self._r.get(self._key(namespace, query_vector, top_k, filter))
        if blob:
            self.hits += 1
            return _deserialize(blob)
        self.misses += 1
        return None

    def put(self, namespace, query_vector, top_k, filter, results):
        self._r.set(
            self._key(namespace, query_vector, top_k, filter),
            _serialize(results),
            ex=self.ttl_seconds,
        )
