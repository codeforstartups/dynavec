"""Unit tests for MistralEmbedder.

The ``mistralai`` SDK is an optional extra, so these tests inject a fake module
into ``sys.modules``: no network, no API key, no dependency on the extra being
installed.
"""

import sys
import types

import pytest

from dynavec.embeddings.mistral import MistralEmbedder
from dynavec.exceptions import MissingDependencyError


def _vec(text, dim):
    """Deterministic stand-in vector so tests can assert batching order."""
    return [float(sum(ord(c) for c in text))] * dim


class _FakeDatum:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddings:
    """Records every create() call instead of talking to Mistral."""

    def __init__(self):
        self.calls = []

    def create(self, model=None, inputs=None, output_dimension=None):
        self.calls.append(
            {"model": model, "inputs": list(inputs), "output_dimension": output_dimension}
        )
        dim = output_dimension or 4
        return _FakeResponse([_FakeDatum(_vec(t, dim), i) for i, t in enumerate(inputs)])


class _FakeMistral:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.embeddings = _FakeEmbeddings()


@pytest.fixture
def fake_mistralai(monkeypatch):
    pkg = types.ModuleType("mistralai")
    client_mod = types.ModuleType("mistralai.client")
    client_mod.Mistral = _FakeMistral
    pkg.client = client_mod
    monkeypatch.setitem(sys.modules, "mistralai", pkg)
    monkeypatch.setitem(sys.modules, "mistralai.client", client_mod)
    return client_mod


def test_import_does_not_require_the_extra():
    # Importing the package must not pull in mistralai; the SDK import lives in
    # __init__ so that this works without `pip install 'dynavec[mistral]'`.
    assert "mistralai" not in sys.modules
    from dynavec.embeddings import MistralEmbedder as Exported

    assert Exported is MistralEmbedder


def test_missing_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "mistralai", None)  # makes the import fail
    monkeypatch.setitem(sys.modules, "mistralai.client", None)
    with pytest.raises(MissingDependencyError) as exc:
        MistralEmbedder()
    assert "pip install 'dynavec[mistral]'" in str(exc.value)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("mistral-embed", 1024),
        ("mistral-embed-2312", 1024),
        ("codestral-embed", 1536),
        ("codestral-embed-2505", 1536),
    ],
)
def test_dimension_from_model_map(fake_mistralai, model, expected):
    assert MistralEmbedder(model=model).dimension == expected


def test_default_model_is_mistral_embed(fake_mistralai):
    emb = MistralEmbedder()
    assert emb.model == "mistral-embed"
    assert emb.dimension == 1024


def test_unknown_model_falls_back_to_1024(fake_mistralai):
    assert MistralEmbedder(model="mistral-embed-does-not-exist").dimension == 1024


def test_explicit_dimension_overrides_map_and_is_sent(fake_mistralai):
    emb = MistralEmbedder(model="codestral-embed", dimension=512)
    assert emb.dimension == 512

    emb.embed_documents(["a"])
    assert emb._client.embeddings.calls[0]["output_dimension"] == 512


def test_output_dimension_omitted_when_not_requested(fake_mistralai):
    emb = MistralEmbedder()
    emb.embed_documents(["a"])
    assert emb._client.embeddings.calls[0]["output_dimension"] is None


def test_api_key_is_passed_through(fake_mistralai):
    assert MistralEmbedder(api_key="sk-test")._client.api_key == "sk-test"


def test_api_key_defers_to_env_var_when_omitted(fake_mistralai):
    # Passing None lets the SDK read MISTRAL_API_KEY itself.
    assert MistralEmbedder()._client.api_key is None


def test_embed_documents_sends_texts_as_inputs(fake_mistralai):
    emb = MistralEmbedder()
    out = emb.embed_documents(["alpha", "beta"])

    assert out == [_vec("alpha", 4), _vec("beta", 4)]
    assert emb._client.embeddings.calls == [
        {"model": "mistral-embed", "inputs": ["alpha", "beta"], "output_dimension": None}
    ]


def test_embed_query_returns_a_single_vector(fake_mistralai):
    # Mistral has no asymmetric query mode, so the base-class default applies.
    emb = MistralEmbedder()
    assert emb.embed_query("alpha") == _vec("alpha", 4)
    assert emb._client.embeddings.calls[0]["inputs"] == ["alpha"]


def test_batching_splits_requests_and_preserves_order(fake_mistralai):
    texts = ["a", "b", "c", "d", "e"]
    emb = MistralEmbedder(batch_size=2)
    out = emb.embed_documents(texts)

    assert [c["inputs"] for c in emb._client.embeddings.calls] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert out == [_vec(t, 4) for t in texts]


def test_empty_input_makes_no_requests(fake_mistralai):
    emb = MistralEmbedder()
    assert emb.embed_documents([]) == []
    assert emb._client.embeddings.calls == []


def test_api_errors_propagate(fake_mistralai):
    class _Boom(_FakeMistral):
        def __init__(self, api_key=None):
            super().__init__(api_key=api_key)

            def _raise(**kwargs):
                raise RuntimeError("rate limited")

            self.embeddings.create = _raise

    fake_mistralai.Mistral = _Boom
    emb = MistralEmbedder()
    with pytest.raises(RuntimeError, match="rate limited"):
        emb.embed_documents(["a"])
