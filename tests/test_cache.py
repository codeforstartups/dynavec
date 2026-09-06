import time
from unittest.mock import patch

import pytest

from dynavec.cache import DynamoDBCache, RedisCache, SemanticCache
from dynavec.config import DynavecConfig
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
