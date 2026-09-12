"""Ollama local embedder (hits local Ollama HTTP API)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import Embedder, Vector

# Known output dimensions for common Ollama embedding models
_MODEL_DIMS = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbedder(Embedder):
    """Embeds text locally using an Ollama server's embedding API.

    Parameters
    ----------
    model:
        Ollama model name, e.g. ``"nomic-embed-text"``.
    host:
        Ollama server base URL. Optional; defaults to the ``OLLAMA_HOST`` environment
        variable or ``"http://localhost:11434"``.
    dimension:
        Optional override for output vector dimension. Inferred from known model defaults
        or first API response if omitted.
    batch_size:
        Number of texts per embedding request. Defaults to 64.
    timeout:
        Request timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        dimension: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        base_host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.host = base_host.rstrip("/")
        self._requested_dim = dimension
        self.dimension = dimension or _MODEL_DIMS.get(model, 768)
        self.batch_size = batch_size
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama embedding request failed ({url}): {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            payload = {"model": self.model, "input": chunk}
            res = self._post("/api/embed", payload)
            embeddings = res.get("embeddings", [])
            out.extend(embeddings)

        if (
            out
            and self._requested_dim is None
            and self.dimension == 768
            and self.model not in _MODEL_DIMS
        ):
            self.dimension = len(out[0])

        return out
