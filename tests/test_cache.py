import fnmatch
import time
from unittest.mock import patch

import pytest

from dynavec.cache import (
    DynamoDBCache,
    RedisCache,
    SemanticCache,
    _glob_escape,
    warm_cache,
)
from dynavec.config import DynavecConfig
from dynavec.exceptions import ConfigurationError
from dynavec.models import SearchResult


def _res(i):
    return [SearchResult(id=i, score=1.0, text=f"doc {i}")]


def test_exact_hit():
    c = SemanticCache(threshold=0.99)
    c.put("ns", [1.0, 0.0], 5, None, _res("a"))
    hit = c.get("ns", [1.0, 0.0], 5, None)
    assert hit and hit[0].id == "a"


def test_near_duplicate_hit_above_threshold():
    c = SemanticCache(threshold=0.9)
    c.put("ns", [1.0, 0.0], 5, None, _res("a"))
    # slightly perturbed query, still very similar in direction
    hit = c.get("ns", [0.99, 0.02], 5, None)
    assert hit and hit[0].id == "a"


def test_dissimilar_miss():
    c = SemanticCache(threshold=0.95)
    c.put("ns", [1.0, 0.0], 5, None, _res("a"))
    assert c.get("ns", [0.0, 1.0], 5, None) is None


def test_namespace_and_topk_scoping():
    c = SemanticCache(threshold=0.5)
    c.put("ns1", [1.0, 0.0], 5, None, _res("a"))
    assert c.get("ns2", [1.0, 0.0], 5, None) is None   # different namespace
    assert c.get("ns1", [1.0, 0.0], 10, None) is None  # different top_k


def test_filter_scoping():
    c = SemanticCache(threshold=0.5)
    c.put("ns", [1.0, 0.0], 5, {"a": 1}, _res("a"))
    assert c.get("ns", [1.0, 0.0], 5, {"a": 2}) is None


def test_lru_eviction():
    c = SemanticCache(threshold=0.999, max_size=2)
    c.put("ns", [1.0, 0.0], 5, None, _res("a"))
    c.put("ns", [0.0, 1.0], 5, None, _res("b"))
    c.put("ns", [0.0, 0.0, 1.0] and [1.0, 1.0], 5, None, _res("c"))  # 3rd -> evict oldest
    total = sum(len(b) for b in c._buckets.values())
    assert total <= 2


def test_semantic_cache_evicts_oldest_entries_to_stay_within_byte_limit():
    probe = SemanticCache()
    probe.put("ns", [1.0, 0.0], 5, None, _res("a"))
    one_entry_bytes = probe.size_bytes

    cache = SemanticCache(threshold=0.999, max_bytes=one_entry_bytes * 2)
    cache.put("first", [1.0, 0.0], 5, None, _res("a"))
    cache.put("second", [0.0, 1.0], 5, None, _res("b"))
    cache.put("third", [1.0, 1.0], 5, None, _res("c"))

    assert cache.size_bytes <= one_entry_bytes * 2
    assert cache.get("first", [1.0, 0.0], 5, None) is None
    assert cache.get("second", [0.0, 1.0], 5, None)
    assert cache.get("third", [1.0, 1.0], 5, None)


def test_semantic_cache_byte_accounting_handles_replacement_and_oversized_entries():
    cache = SemanticCache(max_bytes=10_000)
    cache.put("ns", [1.0, 0.0], 5, None, _res("short"))
    original_size = cache.size_bytes

    cache.put("ns", [1.0, 0.0], 5, None, _res("a much longer result identifier"))

    assert cache.size_bytes > original_size
    assert sum(len(bucket) for bucket in cache._buckets.values()) == 1

    too_small = SemanticCache(max_bytes=1)
    too_small.put("ns", [1.0, 0.0], 5, None, _res("a"))
    assert too_small.size_bytes == 0
    assert too_small.get("ns", [1.0, 0.0], 5, None) is None

    marker = object()
    non_serializable = [SearchResult(id="x", score=1.0, metadata={"marker": marker})]
    cache.put("ns", [0.0, 1.0], 5, None, non_serializable)
    hit = cache.get("ns", [0.0, 1.0], 5, None)
    assert hit and hit[0].metadata["marker"] is marker


def test_semantic_cache_stats():
    c = SemanticCache(threshold=0.9)
    assert c.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}

    # miss on empty
    assert c.get("ns", [1.0, 0.0], 5, None) is None
    assert c.stats() == {"hits": 0, "misses": 1, "hit_rate": 0.0}

    # put and hit
    c.put("ns", [1.0, 0.0], 5, None, _res("a"))
    hit = c.get("ns", [1.0, 0.0], 5, None)
    assert hit and hit[0].id == "a"
    assert c.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}

    # near-duplicate hit
    hit2 = c.get("ns", [0.99, 0.02], 5, None)
    assert hit2 and hit2[0].id == "a"
    assert c.hits == 2
    assert c.misses == 1
    assert c.stats()["hit_rate"] == pytest.approx(2 / 3)

    # dissimilar miss
    assert c.get("ns", [0.0, 1.0], 5, None) is None
    assert c.stats() == {"hits": 2, "misses": 2, "hit_rate": 0.5}

    # reset
    c.reset_stats()
    assert c.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}


def test_dynamodb_cache_stats():
    class FakeTable:
        def __init__(self):
            self.items = {}

        def get_item(self, Key):
            pk = Key["pk"]
            if pk in self.items:
                return {"Item": self.items[pk]}
            return {}

        def put_item(self, Item):
            self.items[Item["pk"]] = Item

    class FakeSession:
        def __init__(self, table):
            self._table = table

        def resource(self, name, region_name=None):
            fake_table = self._table

            class Resource:
                def Table(self, table_name):
                    return fake_table

            return Resource()

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=2)
    fake_table = FakeTable()
    session = FakeSession(fake_table)

    cache = DynamoDBCache(cfg, boto_session=session, ttl_seconds=60)
    assert cache.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}

    # miss on empty
    assert cache.get("ns", [1.0, 0.0], 5, None) is None
    assert cache.stats() == {"hits": 0, "misses": 1, "hit_rate": 0.0}

    # put and hit
    cache.put("ns", [1.0, 0.0], 5, None, _res("a"))
    res = cache.get("ns", [1.0, 0.0], 5, None)
    assert res and res[0].id == "a"
    assert cache.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}

    # expired ttl -> miss
    pk = DynamoDBCache._pk("ns", [1.0, 0.0], 5, None)
    fake_table.items[pk]["ttl"] = int(time.time()) - 100
    assert cache.get("ns", [1.0, 0.0], 5, None) is None
    assert cache.stats() == {"hits": 1, "misses": 2, "hit_rate": pytest.approx(1 / 3)}


def test_dynamodb_cache_ttl_jitter():
    class FakeTable:
        def __init__(self):
            self.items = []

        def get_item(self, Key):
            return {}

        def put_item(self, Item):
            self.items.append(Item)

    class FakeSession:
        def __init__(self, table):
            self._table = table

        def resource(self, name, region_name=None):
            fake_table = self._table

            class Resource:
                def Table(self, table_name):
                    return fake_table

            return Resource()

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=2)
    fake_table = FakeTable()
    session = FakeSession(fake_table)
    cache = DynamoDBCache(
        cfg,
        boto_session=session,
        ttl_seconds=60,
        ttl_jitter_seconds=30,
    )

    with patch("dynavec.cache.time.time", return_value=1_000), patch(
        "dynavec.cache.random.randint", side_effect=[0, 30]
    ) as randint:
        cache.put("ns", [1.0, 0.0], 5, None, _res("a"))
        cache.put("ns", [0.0, 1.0], 5, None, _res("b"))

    assert [item["ttl"] for item in fake_table.items] == [1_060, 1_090]
    assert randint.call_args_list == [((0, 30),), ((0, 30),)]


def test_dynamodb_cache_rejects_negative_ttl_jitter():
    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=2)

    with pytest.raises(ValueError, match="ttl_jitter_seconds must be non-negative"):
        DynamoDBCache(cfg, ttl_jitter_seconds=-1)


def test_redis_cache_stats():
    class FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None):
            self.data[key] = value

    fake_redis = FakeRedis()
    cache = RedisCache(client=fake_redis, ttl_seconds=60)
    assert cache.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}

    # miss
    assert cache.get("ns", [1.0, 0.0], 5, None) is None
    assert cache.stats() == {"hits": 0, "misses": 1, "hit_rate": 0.0}

    # put and hit
    cache.put("ns", [1.0, 0.0], 5, None, _res("b"))
    res = cache.get("ns", [1.0, 0.0], 5, None)
    assert res and res[0].id == "b"
    assert cache.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}

    cache.reset_stats()
    assert cache.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}


def test_warm_prepopulates_cache():
    class FakeClient:
        def __init__(self, cache):
            self.cache = cache
            self.searches = []
            self._vecs = {
                "how do i upsert": [1.0, 0.0],
                "what is a namespace": [0.0, 1.0],
            }

        def search(self, query, *, namespace="default", top_k=10, use_cache=None, **kw):
            vector = self._vecs[query]
            if use_cache:
                cached = self.cache.get(namespace, vector, top_k, None)
                if cached is not None:
                    return cached
            self.searches.append(query)
            results = _res(query)
            if use_cache:
                self.cache.put(namespace, vector, top_k, None, results)
            return results

    cache = SemanticCache()
    client = FakeClient(cache)
    queries = list(client._vecs)

    assert warm_cache(client, queries, top_k=5) == 2
    assert client.searches == queries

    # a repeat of a warmed query is served from the cache, not re-searched
    hit = client.search("how do i upsert", top_k=5, use_cache=True)
    assert hit and hit[0].id == "how do i upsert"
    assert client.searches == queries
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 2


def test_warm_reports_zero_for_empty_queries():
    class FakeClient:
        cache = SemanticCache()

        def search(self, *args, **kwargs):
            raise AssertionError("no queries to search")

    assert warm_cache(FakeClient(), []) == 0


def test_warm_requires_a_cache():
    class FakeClient:
        cache = None

        def search(self, *args, **kwargs):
            raise AssertionError("should not search without a cache")

    with pytest.raises(ConfigurationError, match="cache"):
        warm_cache(FakeClient(), ["what is vector search"])


def test_semantic_cache_invalidate_namespace():
    c = SemanticCache(threshold=0.9)
    c.put("ns-a", [1.0, 0.0], 5, None, _res("a"))
    c.put("ns-b", [1.0, 0.0], 5, None, _res("b"))
    size_before = c.size_bytes
    assert size_before > 0

    c.invalidate("ns-a")

    assert c.get("ns-a", [1.0, 0.0], 5, None) is None
    hit = c.get("ns-b", [1.0, 0.0], 5, None)
    assert hit and hit[0].id == "b"
    assert len(c._buckets) == 1 and len(c._lru) == 1
    assert 0 < c.size_bytes < size_before

    # repeat and unknown-namespace invalidations are no-ops
    c.invalidate("ns-a")
    c.invalidate("ns-c")
    assert c.get("ns-b", [1.0, 0.0], 5, None) is not None

    c.invalidate("ns-b")
    assert c.size_bytes == 0 and not c._buckets and not c._lru and not c._sig_ns


def test_dynamodb_cache_invalidate():
    class FakeTable:
        def __init__(self):
            self.items = {}

        def get_item(self, Key):
            item = self.items.get(Key["pk"])
            return {"Item": item} if item is not None else {}

        def put_item(self, Item):
            self.items[Item["pk"]] = Item

    class FakeSession:
        def __init__(self, table):
            self._table = table

        def resource(self, name, region_name=None):
            fake_table = self._table

            class Resource:
                def Table(self, table_name):
                    return fake_table

            return Resource()

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=2)
    cache = DynamoDBCache(cfg, boto_session=FakeSession(FakeTable()), ttl_seconds=60)

    cache.put("ns", [1.0, 0.0], 5, None, _res("a"))
    cache.put("other", [1.0, 0.0], 5, None, _res("b"))
    assert cache.get("ns", [1.0, 0.0], 5, None)

    cache.invalidate("ns")

    assert cache.get("ns", [1.0, 0.0], 5, None) is None
    assert cache.get("other", [1.0, 0.0], 5, None)[0].id == "b"

    # entries written after the invalidation hit again
    cache.put("ns", [1.0, 0.0], 5, None, _res("a2"))
    assert cache.get("ns", [1.0, 0.0], 5, None)[0].id == "a2"


def test_redis_cache_invalidate():
    class FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None):
            self.data[key] = value

        def delete(self, *keys):
            for k in keys:
                self.data.pop(k, None)

        def scan_iter(self, match=None):
            for k in list(self.data):
                if match is None or fnmatch.fnmatch(k, match):
                    yield k

    fake_redis = FakeRedis()
    cache = RedisCache(client=fake_redis, ttl_seconds=60)
    cache.put("ns", [1.0, 0.0], 5, None, _res("a"))
    cache.put("other", [1.0, 0.0], 5, None, _res("b"))

    cache.invalidate("ns")

    assert cache.get("ns", [1.0, 0.0], 5, None) is None
    assert cache.get("other", [1.0, 0.0], 5, None)[0].id == "b"


def test_glob_escape():
    assert _glob_escape("plain-ns_1") == "plain-ns_1"
    assert _glob_escape("a*b?[c]^d\\e") == "a\\*b\\?\\[c\\]\\^d\\\\e"
