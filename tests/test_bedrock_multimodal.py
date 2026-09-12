"""Unit tests for BedrockTitanMultimodalEmbedder."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock

import pytest

from dynavec.embeddings import BedrockTitanMultimodalEmbedder
from dynavec.embeddings.bedrock import _TITAN_MULTIMODAL_DIMS


def _mock_bedrock_response(dim: int = 1024) -> dict:
    body_payload = {
        "embedding": [0.1] * dim,
        "inputTextTokenCount": 3,
    }
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(body_payload).encode("utf-8")
    return {"body": mock_body}


@pytest.fixture
def mock_boto_session():
    session = MagicMock()
    client = MagicMock()
    session.client.return_value = client
    return session, client


def test_import_and_lazy_export():
    from dynavec.embeddings import BedrockTitanMultimodalEmbedder as Exported

    assert Exported is BedrockTitanMultimodalEmbedder


def test_invalid_dimension_raises():
    for bad_dim in [0, 128, 512, 1536]:
        with pytest.raises(ValueError, match="Invalid dimension"):
            BedrockTitanMultimodalEmbedder(dimension=bad_dim)


@pytest.mark.parametrize("dim", sorted(_TITAN_MULTIMODAL_DIMS))
def test_supported_dimensions(mock_boto_session, dim):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(dim=dim)

    embedder = BedrockTitanMultimodalEmbedder(dimension=dim, boto_session=session)
    assert embedder.dimension == dim

    vec = embedder.embed_query("test query")
    assert len(vec) == dim

    call_args = client.invoke_model.call_args[1]
    payload = json.loads(call_args["body"])
    assert payload["embeddingConfig"]["outputEmbeddingLength"] == dim
    assert payload["inputText"] == "test query"


def test_embed_documents_and_query(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    docs = embedder.embed_documents(["hello", "world"])
    assert len(docs) == 2
    assert len(docs[0]) == 1024

    query_vec = embedder.embed_query("search")
    assert len(query_vec) == 1024
    assert client.invoke_model.call_count == 3


def test_embed_image_from_bytes(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    sample_bytes = b"fake_png_binary_data"
    expected_b64 = base64.b64encode(sample_bytes).decode("ascii")

    vec = embedder.embed_image(sample_bytes)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputImage"] == expected_b64
    assert "inputText" not in payload


def test_embed_image_from_base64_string(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"

    vec = embedder.embed_image(sample_b64)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputImage"] == sample_b64


def test_embed_image_from_file_path(mock_boto_session, tmp_path):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    img_file = tmp_path / "sample.jpg"
    img_bytes = b"\xff\xd8\xff\xe0test_jpeg_bytes"
    img_file.write_bytes(img_bytes)
    expected_b64 = base64.b64encode(img_bytes).decode("ascii")

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_image(img_file)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputImage"] == expected_b64


def test_embed_image_from_binary_stream(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    stream = io.BytesIO(b"streamed_image_bytes")
    expected_b64 = base64.b64encode(b"streamed_image_bytes").decode("ascii")

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_image(stream)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputImage"] == expected_b64


def test_embed_images_batch(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    images = [b"img1", b"img2", b"img3"]

    vecs = embedder.embed_images(images)
    assert len(vecs) == 3
    assert client.invoke_model.call_count == 3


def test_embed_multimodal_combined(mock_boto_session):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    sample_bytes = b"car_photo_bytes"
    expected_b64 = base64.b64encode(sample_bytes).decode("ascii")

    vec = embedder.embed_multimodal(text="red sports car", image=sample_bytes)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputText"] == "red sports car"
    assert payload["inputImage"] == expected_b64


def test_missing_both_text_and_image_raises(mock_boto_session):
    session, _ = mock_boto_session
    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)

    with pytest.raises(ValueError, match="At least one of 'text' or 'image'"):
        embedder.embed_multimodal(text=None, image=None)


def test_unsupported_image_type_raises(mock_boto_session):
    session, _ = mock_boto_session
    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)

    with pytest.raises(TypeError, match="Unsupported image type"):
        embedder.embed_image(12345)


def test_embed_image_from_string_file_path(mock_boto_session, tmp_path):
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    img_file = tmp_path / "photo.png"
    img_bytes = b"\x89PNG_binary"
    img_file.write_bytes(img_bytes)
    expected_b64 = base64.b64encode(img_bytes).decode("ascii")

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_image(str(img_file))
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputImage"] == expected_b64


def test_embed_multimodal_text_only(mock_boto_session):
    """embed_multimodal with text and no image should send only inputText."""
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_multimodal(text="sunset over mountains", image=None)
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputText"] == "sunset over mountains"
    assert "inputImage" not in payload


def test_embed_multimodal_image_only(mock_boto_session):
    """embed_multimodal with image and no text should send only inputImage."""
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_multimodal(text=None, image=b"photo_bytes")
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert "inputText" not in payload
    assert payload["inputImage"] == base64.b64encode(b"photo_bytes").decode("ascii")


def test_empty_string_text_is_sent(mock_boto_session):
    """Empty string text should be sent to the API, not silently dropped."""
    session, client = mock_boto_session
    client.invoke_model.return_value = _mock_bedrock_response(1024)

    embedder = BedrockTitanMultimodalEmbedder(boto_session=session)
    vec = embedder.embed_multimodal(text="", image=b"img_bytes")
    assert len(vec) == 1024

    payload = json.loads(client.invoke_model.call_args[1]["body"])
    assert payload["inputText"] == ""
    assert "inputImage" in payload

