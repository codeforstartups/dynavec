"""Unit tests for HFInferenceEmbedder."""

from unittest.mock import MagicMock, patch
import pytest

from dynavec.embeddings import HFInferenceEmbedder


def test_import_lazy():
    from dynavec.embeddings import HFInferenceEmbedder as Exported
    from dynavec.embeddings.huggingface import HFInferenceEmbedder as Direct

    assert Exported is Direct


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Hugging Face API token is required"):
        HFInferenceEmbedder()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake_hf_token")
    emb = HFInferenceEmbedder()
    assert emb.api_key == "fake_hf_token"
    assert emb.dimension == 384


@patch("requests.post")
def test_embed_documents_batching(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_post.return_value = mock_response

    emb = HFInferenceEmbedder(api_key="fake_token", batch_size=2)
    res = emb.embed_documents(["a", "b", "c"])

    # 3 items with batch size 2 should trigger 2 requests
    assert mock_post.call_count == 2
    assert len(res) == 4


def test_empty_input():
    emb = HFInferenceEmbedder(api_key="fake_token")
    assert emb.embed_documents([]) == []