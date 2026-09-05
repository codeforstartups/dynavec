"""Google Gemini embedder (bring your own GOOGLE_API_KEY)."""

from __future__ import annotations

from typing import Optional

from ..exceptions import MissingDependencyError
from .base import Embedder, Vector

_MODEL_DIMS = {
    "models/text-embedding-004": 768,
    "text-embedding-004": 768,
    "models/embedding-001": 768,
}


class GeminiEmbedder(Embedder):
    """Embeds text with Google's Generative AI embeddings.

    Parameters
    ----------
    model:
        e.g. ``"text-embedding-004"``.
    api_key:
        Optional; falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` env vars.
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("GeminiEmbedder", "google-generativeai", "gemini") from exc

        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.dimension = dimension or _MODEL_DIMS.get(model, 768)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            resp = self._genai.embed_content(
                model=self.model, content=text, task_type="retrieval_document"
            )
            out.append(resp["embedding"])
        return out

    def embed_query(self, text: str) -> Vector:
        resp = self._genai.embed_content(
            model=self.model, content=text, task_type="retrieval_query"
        )
        return resp["embedding"]
