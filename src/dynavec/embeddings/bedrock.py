"""Amazon Bedrock embedder.

Fully in-account: no third-party vendor, credentials are the same AWS creds used
for DynamoDB and S3 Vectors. Great for the compliance-first story.
"""

from __future__ import annotations

import json
from typing import Optional

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
        region: Optional[str] = None,
        dimension: Optional[int] = None,
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
