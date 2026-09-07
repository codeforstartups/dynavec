"""Unit tests for VoyageEmbedder.

The ``voyageai`` SDK is an optional extra, so these tests inject a fake module
into ``sys.modules``: no network, no API key, no dependency on the extra being
installed.
"""

import sys
import types

import pytest

from dynavec.embeddings.voyage import VoyageEmbedder
from dynavec.exceptions import MissingDependencyError


def _vec(text, dim):
    """Deterministic stand-in vector so tests can assert batching order."""
    return [float(sum(ord(c) for c in text))] * dim


class _FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeClient:
    """Records every embed() call instead of talking to Voyage."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.calls = []

    def embed(self, texts, model=None, input_type=None, output_dimension=None):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "output_dimension": output_dimension,
            }
        )
        dim = output_dimension or 4
        return _FakeResponse([_vec(t, dim) for t in texts])


@pytest.fixture
def fake_voyageai(monkeypatch):
    module = types.ModuleType("voyageai")
    module.Client = _FakeClient
    monkeypatch.setitem(sys.modules, "voyageai", module)
    return module


def test_import_does_not_require_the_extra():
    # Importing the package must not pull in voyageai; the SDK import lives in
    # __init__ so that this works without `pip install 'dynavec[voyage]'`.
    assert "voyageai" not in sys.modules
    from dynavec.embeddings import VoyageEmbedder as Exported

    assert Exported is VoyageEmbedder


def test_missing_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "voyageai", None)  # makes `import voyageai` fail
    with pytest.raises(MissingDependencyError) as exc:
        VoyageEmbedder()
    assert "pip install 'dynavec[voyage]'" in str(exc.value)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("voyage-4", 1024),
        ("voyage-4-lite", 1024),
        ("voyage-code-4", 1024),
        ("voyage-3", 1024),
        ("voyage-3-lite", 512),
        ("voyage-code-3", 1024),
        ("voyage-large-2", 1536),
    ],
)
def test_dimension_from_model_map(fake_voyageai, model, expected):
    assert VoyageEmbedder(model=model).dimension == expected


def test_unknown_model_falls_back_to_1024(fake_voyageai):
    assert VoyageEmbedder(model="voyage-does-not-exist").dimension == 1024


def test_explicit_dimension_overrides_map_and_is_sent(fake_voyageai):
    emb = VoyageEmbedder(model="voyage-4", dimension=256)
    assert emb.dimension == 256

    emb.embed_documents(["a"])
    assert emb._client.calls[0]["output_dimension"] == 256


def test_output_dimension_omitted_when_not_requested(fake_voyageai):
    emb = VoyageEmbedder(model="voyage-4")
    emb.embed_documents(["a"])
    assert emb._client.calls[0]["output_dimension"] is None


def test_api_key_is_passed_through(fake_voyageai):
    assert VoyageEmbedder(api_key="sk-test")._client.api_key == "sk-test"


def test_api_key_defers_to_env_var_when_omitted(fake_voyageai):
    # Passing None lets the SDK read VOYAGE_API_KEY itself.
    assert VoyageEmbedder()._client.api_key is None


def test_embed_documents_uses_document_input_type(fake_voyageai):
    emb = VoyageEmbedder(model="voyage-4")
    out = emb.embed_documents(["alpha", "beta"])

    assert out == [_vec("alpha", 4), _vec("beta", 4)]
    assert emb._client.calls == [
        {
            "texts": ["alpha", "beta"],
            "model": "voyage-4",
            "input_type": "document",
            "output_dimension": None,
        }
    ]


def test_embed_query_uses_query_input_type_and_returns_one_vector(fake_voyageai):
    emb = VoyageEmbedder(model="voyage-4")
    out = emb.embed_query("alpha")

    assert out == _vec("alpha", 4)
    assert emb._client.calls[0]["input_type"] == "query"
    assert emb._client.calls[0]["texts"] == ["alpha"]


def test_batching_splits_requests_and_preserves_order(fake_voyageai):
    texts = ["a", "b", "c", "d", "e"]
    emb = VoyageEmbedder(model="voyage-4", batch_size=2)
    out = emb.embed_documents(texts)

    assert [c["texts"] for c in emb._client.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert out == [_vec(t, 4) for t in texts]


def test_empty_input_makes_no_requests(fake_voyageai):
    emb = VoyageEmbedder(model="voyage-4")
    assert emb.embed_documents([]) == []
    assert emb._client.calls == []


def test_api_errors_propagate(fake_voyageai):
    class _Boom(_FakeClient):
        def embed(self, texts, **kwargs):
            raise RuntimeError("rate limited")

    fake_voyageai.Client = _Boom
    emb = VoyageEmbedder(model="voyage-4")
    with pytest.raises(RuntimeError, match="rate limited"):
        emb.embed_documents(["a"])
