"""CachedEmbedder — wraps any Embedder with an EmbeddingCache.

Usage::

    from dynavec.embeddings import OpenAIEmbedder, CachedEmbedder
    from dynavec.embeddings.cache import InMemoryCache

    embedder = CachedEmbedder(
        OpenAIEmbedder(model="text-embedding-3-small"),
        cache=InMemoryCache(max_size=50_000),
    )
    # First call → hits OpenAI API
    vecs = embedder.embed_documents(["hello world"])
    # Second call → cache hit, no API call
    vecs = embedder.embed_documents(["hello world"])
"""

from __future__ import annotations

from .base import Embedder, Vector
from .cache import EmbeddingCache, InMemoryCache


class CachedEmbedder(Embedder):
    """Decorator that adds caching to any :class:`Embedder`.

    Parameters
    ----------
    inner:
        The real embedding backend to wrap.
    cache:
        Any :class:`~dynavec.embeddings.cache.EmbeddingCache`.
        Defaults to a fresh ``InMemoryCache(max_size=10_000)``.
    model_key:
        String used as the model component of the cache key.
        Defaults to ``type(inner).__name__`` — so OpenAIEmbedder and
        GeminiEmbedder never share cache entries even for identical text.
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        cache: EmbeddingCache | None = None,
        model_key: str | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else InMemoryCache()
        self._model_key = model_key or type(inner).__name__
        self.dimension = inner.dimension

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        results: list[Vector | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        # --- split into hits and misses ---
        for i, text in enumerate(texts):
            vec = self._cache.get_for(text, self._model_key)
            if vec is not None:
                results[i] = vec
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # --- embed only misses (one batched API call) ---
        if miss_texts:
            fresh = self._inner.embed_documents(miss_texts)
            for text, vec in zip(miss_texts, fresh):
                self._cache.set_for(text, self._model_key, vec)
            for idx, vec in zip(miss_indices, fresh):
                results[idx] = vec

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> Vector:
        vec = self._cache.get_for(text, self._model_key)
        if vec is not None:
            return vec
        vec = self._inner.embed_query(text)
        self._cache.set_for(text, self._model_key, vec)
        return vec

    async def aembed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        results: list[Vector | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            vec = self._cache.get_for(text, self._model_key)
            if vec is not None:
                results[i] = vec
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh = await self._inner.aembed_documents(miss_texts)
            for text, vec in zip(miss_texts, fresh):
                self._cache.set_for(text, self._model_key, vec)
            for idx, vec in zip(miss_indices, fresh):
                results[idx] = vec

        return results  # type: ignore[return-value]

    async def aembed_query(self, text: str) -> Vector:
        vec = self._cache.get_for(text, self._model_key)
        if vec is not None:
            return vec
        vec = await self._inner.aembed_query(text)
        self._cache.set_for(text, self._model_key, vec)
        return vec

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def inner(self) -> Embedder:
        return self._inner

    def __repr__(self) -> str:
        return (
            f"CachedEmbedder(inner={self._inner!r}, "
            f"model_key={self._model_key!r})"
        )
