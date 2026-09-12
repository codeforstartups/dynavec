"""Embedder abstraction.

An :class:`Embedder` turns text into float vectors. dynavec never requires one:
users may upsert/query with pre-computed vectors instead. When they do want
dynavec to embed, they choose a backend (OpenAI, Gemini, Cohere, Bedrock,
sentence-transformers, ...) and supply their *own* API key / credentials.
"""

from __future__ import annotations

import asyncio
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

    async def aembed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of documents asynchronously.

        The default implementation delegates to synchronous :meth:`embed_documents`
        inside a worker thread using :func:`asyncio.to_thread`. Subclasses with native
        asynchronous client support should override this method directly.

        Parameters
        ----------
        texts:
            A list of text strings to embed.

        Returns
        -------
        list[Vector]
            A list of float vector embeddings corresponding to the input texts.
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> Vector:
        """Embed a single query string asynchronously.

        The default implementation delegates to synchronous :meth:`embed_query`
        inside a worker thread using :func:`asyncio.to_thread`. Subclasses with native
        asymmetric or async query support should override this method directly.

        Parameters
        ----------
        text:
            A query string to embed.

        Returns
        -------
        Vector
            A single float vector embedding.
        """
        return await asyncio.to_thread(self.embed_query, text)

    def __call__(self, texts: list[str]) -> list[Vector]:
        return self.embed_documents(texts)
