"""Tests for the SemanticCache (pure, no AWS)."""

from dynavec.cache import SemanticCache
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
