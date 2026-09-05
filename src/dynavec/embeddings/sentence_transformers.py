"""Local, free embedder via sentence-transformers (no API key, runs in-process)."""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from .base import Embedder, Vector


class SentenceTransformerEmbedder(Embedder):
    """Embeds text with a local sentence-transformers model.

    Zero external calls — useful for offline dev, tests, air-gapped compliance
    setups, and cost-free benchmarking.

    Parameters
    ----------
    model:
        e.g. ``"all-MiniLM-L6-v2"`` (384-dim) or ``"BAAI/bge-small-en-v1.5"``.
    device:
        ``"cpu"``, ``"cuda"``, ... passed straight to SentenceTransformer.
    normalize:
        L2-normalize outputs (recommended with cosine indexes).
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError(
                "SentenceTransformerEmbedder", "sentence-transformers", "sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        self.normalize = normalize
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        arr = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return arr.tolist()
