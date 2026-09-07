"""Voyage AI embedder (bring your own VOYAGE_API_KEY)."""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from .base import Embedder, Vector

# Known output dimensions for common models (used when dimension is not given).
_MODEL_DIMS = {
    "voyage-4-large": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-code-4": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-multilingual-2": 1024,
    "voyage-large-2": 1536,
    "voyage-2": 1024,
}


class VoyageEmbedder(Embedder):
    """Embeds text with Voyage AI's embeddings API.

    Parameters
    ----------
    model:
        Model id, e.g. ``"voyage-4"``.
    api_key:
        Optional; falls back to the ``VOYAGE_API_KEY`` environment variable.
    dimension:
        Optional override. The voyage-4-*, voyage-3.5-* and voyage-code-3 models
        support 256/512/1024/2048 via the API's ``output_dimension`` parameter;
        pass it here to request a different vector size.
    batch_size:
        Texts per request. The API caps a single call at 1000 texts; the default
        stays well under that to also respect the per-request token limit.
    """

    def __init__(
        self,
        model: str = "voyage-4",
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("VoyageEmbedder", "voyageai", "voyage") from exc

        self.model = model
        self._client = voyageai.Client(api_key=api_key)
        self._requested_dim = dimension
        self.dimension = dimension or _MODEL_DIMS.get(model, 1024)
        self.batch_size = batch_size

    def _embed(self, texts: list[str], input_type: str) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            kwargs = {"model": self.model, "input_type": input_type}
            if self._requested_dim is not None:
                kwargs["output_dimension"] = self._requested_dim
            resp = self._client.embed(chunk, **kwargs)
            out.extend(resp.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "query")[0]
