"""Tests for EmbeddingCache and CachedEmbedder."""

from __future__ import annotations

import asyncio
import hashlib
from typing import ClassVar

import pytest

from dynavec.embeddings.base import Embedder, Vector
from dynavec.embeddings.cache import EmbeddingCache, InMemoryCache, _make_key
from dynavec.embeddings.cached import CachedEmbedder

# ---------------------------------------------------------------------------
# Fake embedders (same style as test_async_embeddings.py)
# ---------------------------------------------------------------------------

class _CountingEmbedder(Embedder):
    """Deterministic embedder that tracks every text it actually embeds."""

    dimension: ClassVar[int] = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []   # one entry per embed_documents call

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> Vector:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(4)]

    @property
    def total_texts_embedded(self) -> int:
        return sum(len(c) for c in self.calls)


class _NativeAsyncEmbedder(Embedder):
    """Embedder with native async — sync path must never be called."""

    dimension: ClassVar[int] = 4

    def __init__(self) -> None:
        self.async_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        raise NotImplementedError("sync path must not be called")

    async def aembed_documents(self, texts: list[str]) -> list[Vector]:
        self.async_calls.append(list(texts))
        await asyncio.sleep(0)
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


# ---------------------------------------------------------------------------
# Custom cache backend (proves pluggability)
# ---------------------------------------------------------------------------

class _MinimalCache(EmbeddingCache):
    """Bare-minimum custom cache to verify the ABC contract works."""

    def __init__(self) -> None:
        super().__init__()
        self._d: dict[str, Vector] = {}

    def get(self, key: str) -> Vector | None:
        return self._d.get(key)

    def set(self, key: str, vector: Vector) -> None:
        self._d[key] = vector


# ---------------------------------------------------------------------------
# 1. _make_key
# ---------------------------------------------------------------------------

def test_make_key_same_inputs_same_key():
    assert _make_key("hello", "m1") == _make_key("hello", "m1")

def test_make_key_different_text_different_key():
    assert _make_key("hello", "m1") != _make_key("world", "m1")

def test_make_key_different_model_different_key():
    assert _make_key("hello", "m1") != _make_key("hello", "m2")

def test_make_key_format():
    key = _make_key("abc", "my-model")
    sha, model = key.split(":")
    assert len(sha) == 64        # sha256 hex is always 64 chars
    assert model == "my-model"


# ---------------------------------------------------------------------------
# 2. InMemoryCache
# ---------------------------------------------------------------------------

def test_inmemory_miss_returns_none():
    assert InMemoryCache().get("nonexistent") is None

def test_inmemory_set_then_get():
    cache = InMemoryCache()
    cache.set("k", [1.0, 2.0])
    assert cache.get("k") == [1.0, 2.0]

def test_inmemory_get_for_set_for():
    cache = InMemoryCache()
    cache.set_for("hello", "model-a", [0.5, 0.5])
    assert cache.get_for("hello", "model-a") == [0.5, 0.5]
    assert cache.get_for("hello", "model-b") is None   # different model → miss

def test_inmemory_lru_eviction():
    cache = InMemoryCache(max_size=2)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.get("a")             # touch "a" → now "b" is LRU
    cache.set("c", [3.0])     # should evict "b", not "a"
    assert cache.get("b") is None
    assert cache.get("a") == [1.0]
    assert cache.get("c") == [3.0]

def test_inmemory_max_size_none_is_unlimited():
    cache = InMemoryCache(max_size=None)
    for i in range(10_000):
        cache.set(str(i), [float(i)])
    assert len(cache) == 10_000

def test_inmemory_negative_max_size_raises():
    with pytest.raises(ValueError):
        InMemoryCache(max_size=-1)

def test_inmemory_len_and_contains():
    cache = InMemoryCache()
    assert len(cache) == 0
    cache.set("x", [0.0])
    assert len(cache) == 1
    assert "x" in cache
    assert "y" not in cache

def test_inmemory_clear():
    cache = InMemoryCache()
    cache.set("x", [0.0])
    cache.clear()
    assert len(cache) == 0

def test_inmemory_stats_hit_rate():
    cache = InMemoryCache()
    cache.set_for("hi", "m", [1.0])
    cache.get_for("hi", "m")      # hit
    cache.get_for("bye", "m")     # miss
    s = cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["hit_rate"] == 0.5

def test_inmemory_reset_stats():
    cache = InMemoryCache()
    cache.set_for("hi", "m", [1.0])
    cache.get_for("hi", "m")
    cache.reset_stats()
    assert cache.stats() == {"hits": 0, "misses": 0, "hit_rate": 0.0}


# ---------------------------------------------------------------------------
# 3. CachedEmbedder — embed_documents
# ---------------------------------------------------------------------------

def test_first_call_hits_inner():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    embedder.embed_documents(["a", "b"])
    assert inner.calls == [["a", "b"]]

def test_second_call_is_full_cache_hit():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v1 = embedder.embed_documents(["a", "b"])
    v2 = embedder.embed_documents(["a", "b"])
    assert len(inner.calls) == 1    # inner called only once
    assert v1 == v2

def test_partial_hit_calls_inner_only_for_misses():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    embedder.embed_documents(["a"])           # warm "a"
    embedder.embed_documents(["a", "b"])      # "a" hit, "b" miss
    # second call should only have sent "b" to inner
    assert inner.calls[-1] == ["b"]

def test_order_preserved_with_mixed_hits_misses():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v_xyz = embedder.embed_documents(["x", "y", "z"])
    # ask in different order; all are cached now
    v_zyx = embedder.embed_documents(["z", "y", "x"])
    assert v_zyx[0] == v_xyz[2]
    assert v_zyx[1] == v_xyz[1]
    assert v_zyx[2] == v_xyz[0]

def test_empty_input_returns_empty():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    assert embedder.embed_documents([]) == []
    assert inner.calls == []

def test_dimension_delegated_to_inner():
    inner = _CountingEmbedder()
    assert CachedEmbedder(inner).dimension == inner.dimension


# ---------------------------------------------------------------------------
# 4. CachedEmbedder — embed_query
# ---------------------------------------------------------------------------

def test_embed_query_miss_then_hit():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v1 = embedder.embed_query("hello")
    v2 = embedder.embed_query("hello")
    assert v1 == v2
    assert inner.total_texts_embedded == 1   # embedded only once


# ---------------------------------------------------------------------------
# 5. Async variants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aembed_documents_caches():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v1 = await embedder.aembed_documents(["p", "q"])
    v2 = await embedder.aembed_documents(["p", "q"])
    assert len(inner.calls) == 1
    assert v1 == v2

@pytest.mark.asyncio
async def test_aembed_documents_partial_hit():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    await embedder.aembed_documents(["p"])
    await embedder.aembed_documents(["p", "q"])
    assert inner.calls[-1] == ["q"]

@pytest.mark.asyncio
async def test_aembed_query_caches():
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v1 = await embedder.aembed_query("async-test")
    v2 = await embedder.aembed_query("async-test")
    assert v1 == v2
    assert inner.total_texts_embedded == 1

@pytest.mark.asyncio
async def test_aembed_documents_native_async_inner():
    """Cache should work when inner has a native async path."""
    inner = _NativeAsyncEmbedder()
    embedder = CachedEmbedder(inner, cache=InMemoryCache())
    v1 = await embedder.aembed_documents(["foo", "bar"])
    v2 = await embedder.aembed_documents(["foo", "bar"])
    assert len(inner.async_calls) == 1    # native async called only once
    assert v1 == v2


# ---------------------------------------------------------------------------
# 6. model_key isolation
# ---------------------------------------------------------------------------

def test_different_model_keys_dont_share_entries():
    shared_cache = InMemoryCache()
    inner_a = _CountingEmbedder()
    inner_b = _CountingEmbedder()
    ea = CachedEmbedder(inner_a, cache=shared_cache, model_key="model-A")
    eb = CachedEmbedder(inner_b, cache=shared_cache, model_key="model-B")
    ea.embed_documents(["hello"])
    eb.embed_documents(["hello"])     # same text, different key → must be a miss
    assert inner_a.total_texts_embedded == 1
    assert inner_b.total_texts_embedded == 1   # inner_b was also called


# ---------------------------------------------------------------------------
# 7. Pluggable custom backend
# ---------------------------------------------------------------------------

def test_custom_cache_backend_is_used():
    custom = _MinimalCache()
    inner = _CountingEmbedder()
    embedder = CachedEmbedder(inner, cache=custom)
    embedder.embed_documents(["test"])
    embedder.embed_documents(["test"])
    assert inner.total_texts_embedded == 1
    assert embedder.cache is custom

def test_default_cache_is_inmemory():
    embedder = CachedEmbedder(_CountingEmbedder())
    assert isinstance(embedder.cache, InMemoryCache)


@pytest.mark.parametrize("method", ["embed_documents", "embed_query"])
def test_cached_dimension_tracks_ollama_inference(monkeypatch, method):
    from dynavec.embeddings.ollama import OllamaEmbedder

    inner = OllamaEmbedder(model="custom-model")
    monkeypatch.setattr(inner, "_post", lambda *args: {"embeddings": [[0.1, 0.2, 0.3]]})
    cached = CachedEmbedder(inner)
    assert cached.dimension == 768
    getattr(cached, method)(["hello"] if method == "embed_documents" else "hello")
    assert inner.dimension == 3
    assert cached.dimension == 3
