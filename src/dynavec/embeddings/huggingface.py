"""Hugging Face Inference API embedder."""

from __future__ import annotations

import os
from collections.abc import Sequence

import requests

from .base import Embedder, Vector

_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class HFInferenceEmbedder(Embedder):
    """Embeds text using Hugging Face Serverless Inference API."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        api_key: str | None = None,
        batch_size: int = 32,
        dimension: int | None = None,
    ) -> None:
        """Initialize HFInferenceEmbedder."""
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.batch_size = batch_size
        self.dimension = dimension or _MODEL_DIMS.get(model_name, 384)

        if not self.api_key:
            raise ValueError(
                "Hugging Face API token is required. Pass api_key or set HF_TOKEN environment variable."
            )

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a list of documents using HF Inference API with batching."""
        if not texts:
            return []

        out: list[Vector] = []

        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i : i + self.batch_size])

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": chunk, "options": {"wait_for_model": True}},
            )

            if response.status_code != 200:
                if response.status_code in (401, 403):
                    raise ValueError(
                        "Invalid Hugging Face API token or unauthorized request."
                    )
                elif response.status_code == 404:
                    raise ValueError(
                        f"Model '{self.model_name}' not found on Hugging Face Hub."
                    )
                else:
                    raise RuntimeError(
                        f"Hugging Face API request failed ({response.status_code}): {response.text}"
                    )

            data = response.json()
            out.extend(data)

        return out

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string."""
        if not text:
            return []
        results = self.embed_documents([text])
        return results[0] if results else []
