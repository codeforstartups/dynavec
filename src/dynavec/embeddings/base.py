"""Embedder abstraction.

An :class:`Embedder` turns text into float vectors. dynavec never requires one:
users may upsert/query with pre-computed vectors instead. When they do want
dynavec to embed, they choose a backend (OpenAI, Gemini, Cohere, Bedrock,
sentence-transformers, ...) and supply their *own* API key / credentials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

Vector = list[float]


class Embedder(ABC):
    """Base class for all embedding backends."""

    #: Output dimension of this embedder. Must match the S3 Vectors index.
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of documents (indexing side)."""

    def embed_query(self, text: str) -> Vector:
        """Embed a single query. Override if the backend has an asymmetric
        query mode; the default reuses :meth:`embed_documents`."""
        return self.embed_documents([text])[0]

    def __call__(self, texts: list[str]) -> list[Vector]:
        return self.embed_documents(texts)
