"""Mistral embedder (bring your own MISTRAL_API_KEY)."""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from .base import Embedder, Vector

# Known output dimensions for common models (used when dimension is not given).
_MODEL_DIMS = {
    "mistral-embed": 1024,
    "mistral-embed-2312": 1024,
    "codestral-embed": 1536,
    "codestral-embed-2505": 1536,
}


class MistralEmbedder(Embedder):
    """Embeds text with Mistral's embeddings API.

    Parameters
    ----------
    model:
        Model id, e.g. ``"mistral-embed"``.
    api_key:
        Optional; falls back to the ``MISTRAL_API_KEY`` environment variable.
    dimension:
        Optional override. ``codestral-embed`` supports any size up to 3072 via
        the API's ``output_dimension`` parameter; ``mistral-embed`` is fixed at
        1024.
    batch_size:
        Texts per request.
    """

    def __init__(
        self,
        model: str = "mistral-embed",
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            from mistralai.client import Mistral
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("MistralEmbedder", "mistralai", "mistral") from exc

        self.model = model
        self._client = Mistral(api_key=api_key)
        self._requested_dim = dimension
        self.dimension = dimension or _MODEL_DIMS.get(model, 1024)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            kwargs = {"model": self.model, "inputs": chunk}
            if self._requested_dim is not None:
                kwargs["output_dimension"] = self._requested_dim
            resp = self._client.embeddings.create(**kwargs)
            out.extend(d.embedding for d in resp.data)
        return out
