"""OpenAI embedder (bring your own OPENAI_API_KEY)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse

from ..exceptions import MissingDependencyError
from ..utils import retry
from .base import Embedder, Vector

# Known output dimensions for common models (used when dimension is not given).
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

def _is_openai_retryable(exc: Exception) -> bool:
    """Return True for OpenAI rate-limit and server errors."""
    status_code = getattr(exc, "status_code", None)

    return status_code == 429 or (
        isinstance(status_code, int) and 500 <= status_code < 600
    )

def _openai_retry_after(exc: Exception) -> float | None:
    """Return the server-provided Retry-After delay, if available."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if not headers:
        return None

    value = headers.get("retry-after")

    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


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
            resp = self._create_embeddings(
                model=self.model, input=chunk, dimensions=self._requested_dim
            )
            out.extend(d.embedding for d in resp.data)
        return out


    @retry(
        retry_on=_is_openai_retryable,
        retry_delay=_openai_retry_after,
    )
    def _create_embeddings(
        self, *, model: str, input: list[str], dimensions: int | None = None
    ) -> CreateEmbeddingResponse:
        if dimensions is None:
            return self._client.embeddings.create(model=model, input=input)
        return self._client.embeddings.create(
            model=model, input=input, dimensions=dimensions
        )
