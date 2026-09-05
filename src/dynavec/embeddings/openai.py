"""OpenAI embedder (bring your own OPENAI_API_KEY)."""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from .base import Embedder, Vector

# Known output dimensions for common models (used when dimension is not given).
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Embeds text with OpenAI's embeddings API.

    Parameters
    ----------
    model:
        Model id, e.g. ``"text-embedding-3-small"``.
    api_key:
        Optional; falls back to the ``OPENAI_API_KEY`` environment variable.
    dimension:
        Optional override. text-embedding-3-* support shortening via the API's
        ``dimensions`` parameter; pass it here to request a smaller vector.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int | None = None,
        base_url: str | None = None,
        batch_size: int = 256,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("OpenAIEmbedder", "openai", "openai") from exc

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._requested_dim = dimension
        self.dimension = dimension or _MODEL_DIMS.get(model, 1536)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            kwargs = {"model": self.model, "input": chunk}
            if self._requested_dim is not None:
                kwargs["dimensions"] = self._requested_dim
            resp = self._client.embeddings.create(**kwargs)
            out.extend(d.embedding for d in resp.data)
        return out
