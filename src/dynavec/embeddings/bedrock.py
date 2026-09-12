"""Amazon Bedrock embedder.

Fully in-account: no third-party vendor, credentials are the same AWS creds used
for DynamoDB and S3 Vectors. Great for the compliance-first story.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import BinaryIO

from .base import Embedder, Vector

_MODEL_DIMS = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}


class BedrockEmbedder(Embedder):
    """Embeds text with Amazon Bedrock foundation models via ``invoke_model``.

    Parameters
    ----------
    model_id:
        e.g. ``"amazon.titan-embed-text-v2:0"``.
    region:
        AWS region for the ``bedrock-runtime`` client.
    dimension:
        Titan v2 supports 256/512/1024; pass to request a specific size.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str | None = None,
        dimension: int | None = None,
        boto_session=None,
    ) -> None:
        import boto3  # local import keeps base import cheap

        session = boto_session or boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self._requested_dim = dimension
        self.dimension = dimension or _MODEL_DIMS.get(model_id, 1024)
        self._is_titan = model_id.startswith("amazon.titan")
        self._is_cohere = model_id.startswith("cohere.")

    def _invoke(self, text: str, input_type: str) -> Vector:
        if self._is_titan:
            body = {"inputText": text}
            if self._requested_dim is not None:
                body["dimensions"] = self._requested_dim
        elif self._is_cohere:
            body = {"texts": [text], "input_type": input_type}
        else:
            body = {"inputText": text}

        resp = self._client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        if self._is_cohere:
            return payload["embeddings"][0]
        return payload["embedding"]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._invoke(t, "search_document") for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._invoke(text, "search_query")


_TITAN_MULTIMODAL_DIMS = {256, 384, 1024}


class BedrockTitanMultimodalEmbedder(Embedder):
    """Embeds text, images, or multimodal inputs using Amazon Bedrock Titan Multimodal.

    Model: ``amazon.titan-embed-image-v1``

    Maps text and images into a single shared vector space for cross-modal search
    (Text-to-Image, Image-to-Image, Image-to-Text).

    Parameters
    ----------
    dimension:
        Output embedding length: 256, 384, or 1024 (default is 1024).
    region:
        AWS region for the ``bedrock-runtime`` client.
    boto_session:
        Optional pre-configured boto3 Session.
    """

    def __init__(
        self,
        dimension: int = 1024,
        region: str | None = None,
        boto_session=None,
    ) -> None:
        import boto3

        if dimension not in _TITAN_MULTIMODAL_DIMS:
            raise ValueError(
                f"Invalid dimension {dimension} for Titan Multimodal. "
                f"Supported dimensions: {sorted(_TITAN_MULTIMODAL_DIMS)}"
            )

        session = boto_session or boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=region)
        self.model_id = "amazon.titan-embed-image-v1"
        self.dimension = dimension

    def _to_base64(self, image: bytes | str | Path | BinaryIO) -> str:
        """Convert an image (file path, raw bytes, or base64 str) to base64."""
        if isinstance(image, str):
            path = Path(image)
            if path.exists() and path.is_file():
                raw_bytes = path.read_bytes()
            else:
                return image
        elif isinstance(image, Path):
            raw_bytes = image.read_bytes()
        elif isinstance(image, bytes):
            raw_bytes = image
        elif hasattr(image, "read"):
            raw_bytes = image.read()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected bytes, str, or Path.")
        return base64.b64encode(raw_bytes).decode("ascii")

    def _invoke(
        self,
        text: str | None = None,
        image_base64: str | None = None,
    ) -> Vector:
        if text is None and image_base64 is None:
            raise ValueError("At least one of 'text' or 'image' must be provided.")

        body: dict = {
            "embeddingConfig": {
                "outputEmbeddingLength": self.dimension,
            }
        }
        if text is not None:
            body["inputText"] = text
        if image_base64 is not None:
            body["inputImage"] = image_base64

        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
        )
        payload = json.loads(resp["body"].read())
        return payload["embedding"]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of text documents."""
        return [self._invoke(text=t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        """Embed a text query for cross-modal search against image or text vectors."""
        return self._invoke(text=text)

    def embed_image(self, image: bytes | str | Path | BinaryIO) -> Vector:
        """Embed a single image from raw bytes, base64 string, or file path."""
        b64 = self._to_base64(image)
        return self._invoke(image_base64=b64)

    def embed_images(self, images: list[bytes | str | Path | BinaryIO]) -> list[Vector]:
        """Embed a batch of images."""
        return [self.embed_image(img) for img in images]

    def embed_multimodal(
        self,
        text: str | None = None,
        image: bytes | str | Path | BinaryIO | None = None,
    ) -> Vector:
        """Embed combined text and visual content into a joint multimodal vector."""
        b64 = self._to_base64(image) if image is not None else None
        return self._invoke(text=text, image_base64=b64)
