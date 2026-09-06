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
import random
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

    def __init__(self, threshold: float = 0.97, max_size: int = 2048) -> None:
        super().__init__()
        self.threshold = threshold
        self.max_size = max_size
        # key: signature -> OrderedDict[vec_key -> (unit_vec, results)]
        self._buckets: dict[str, OrderedDict[str, tuple]] = {}

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
        for vk, (vec, _res) in bucket.items():
            sim = float(vec @ q)
            if sim > best_sim:
                best_sim, best_key = sim, vk
        if best_key is not None and best_sim >= self.threshold:
            vec, res = bucket.pop(best_key)
            bucket[best_key] = (vec, res)  # move to MRU
            self.hits += 1
            return res
        self.misses += 1
        return None

    def put(self, namespace, query_vector, top_k, filter, results):
        sig = _signature(namespace, top_k, filter)
        bucket = self._buckets.setdefault(sig, OrderedDict())
        vk = _vec_key(query_vector)
        bucket[vk] = (self._unit(np.asarray(query_vector, dtype=np.float32)), results)
        bucket.move_to_end(vk)
        while sum(len(b) for b in self._buckets.values()) > self.max_size:
            # evict the globally oldest entry
            for b in self._buckets.values():
                if b:
                    b.popitem(last=False)
                    break


class DynamoDBCache(BaseCache):
    """Exact-match cache persisted in the dynavec table with TTL expiry.

    Enable DynamoDB TTL on the ``ttl`` attribute of your table for automatic
    eviction (items are also treated as expired client-side as a safety net).
    ``ttl_jitter_seconds`` adds a random delay of up to the configured number
    of seconds to each expiry, spreading simultaneous writes across an expiry
    window to reduce cache stampedes.
    """

    def __init__(
        self,
        config,
        boto_session=None,
        ttl_seconds: int = 3600,
        ttl_jitter_seconds: int = 0,
    ) -> None:
        super().__init__()
        import boto3

        if ttl_jitter_seconds < 0:
            raise ValueError("ttl_jitter_seconds must be non-negative")

        session = boto_session or boto3.Session()
        self._table = session.resource("dynamodb", region_name=config.region).Table(config.table)
        self.ttl_seconds = ttl_seconds
        self.ttl_jitter_seconds = ttl_jitter_seconds

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
        jitter = random.randint(0, self.ttl_jitter_seconds)
        self._table.put_item(
            Item={
                "pk": self._pk(namespace, query_vector, top_k, filter),
                "kind": "querycache",
                "results": _serialize(results),
                "ttl": int(time.time()) + self.ttl_seconds + jitter,
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
