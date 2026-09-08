"""Tests for the ``list_vectors`` maintenance iterator.

Covers the raw ``ListVectors`` page generator on the store and the
namespace-scoped, document-hydrating generator on the client. No network: the
boto3 client is either a ``MagicMock`` paginator (matching the convention in
``test_s3vectors.py``) or a real botocore ``Stubber``, which additionally
validates our request parameters against the actual s3vectors service model.
"""

from unittest.mock import MagicMock

import boto3
import pytest
from botocore.stub import Stubber

from dynavec.client import Dynavec
from dynavec.config import NS_METADATA_KEY, TEXT_METADATA_KEY, DynavecConfig
from dynavec.stores.s3vectors import S3VectorsStore


def _config(**kwargs):
    defaults = dict(
        vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
    )
    defaults.update(kwargs)
    return DynavecConfig(**defaults)


def _store_with_pages(pages, config=None):
    store = S3VectorsStore.__new__(S3VectorsStore)
    store._config = config or _config()
    store._client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    store._client.get_paginator.return_value = paginator
    return store, paginator


# --------------------------------------------------------------- store: pages
def test_list_pages_multi_page_stitched_in_order():
    pages = [
        {"vectors": [{"key": f"k{i}"} for i in range(3)], "nextToken": "t1"},
        {"vectors": [{"key": f"k{i}"} for i in range(3, 6)], "nextToken": "t2"},
        {"vectors": [{"key": f"k{i}"} for i in range(6, 8)]},
    ]
    store, _ = _store_with_pages(pages)

    got = list(store.list_pages())

    assert [len(p) for p in got] == [3, 3, 2]
    assert [v["key"] for p in got for v in p] == [f"k{i}" for i in range(8)]
    store._client.get_paginator.assert_called_once_with("list_vectors")


def test_list_pages_single_page_without_token():
    store, _ = _store_with_pages([{"vectors": [{"key": "k0"}, {"key": "k1"}]}])

    got = list(store.list_pages())

    assert [len(p) for p in got] == [2]


def test_list_pages_empty_index_yields_nothing():
    store, _ = _store_with_pages([{"vectors": []}])

    assert list(store.list_pages()) == []


def test_list_pages_skips_empty_pages():
    pages = [
        {"vectors": [{"key": "k0"}], "nextToken": "t1"},
        {"vectors": [], "nextToken": "t2"},
        {"vectors": [{"key": "k1"}]},
    ]
    store, _ = _store_with_pages(pages)

    assert [v["key"] for p in store.list_pages() for v in p] == ["k0", "k1"]


def test_list_pages_passes_request_params():
    store, paginator = _store_with_pages([{"vectors": [{"key": "k0"}]}])

    list(store.list_pages(return_data=True, return_metadata=True, page_size=250))

    kwargs = paginator.paginate.call_args.kwargs
    assert kwargs["vectorBucketName"] == "test-bucket"
    assert kwargs["indexName"] == "test-index"
    assert kwargs["returnData"] is True
    assert kwargs["returnMetadata"] is True
    assert kwargs["PaginationConfig"] == {"PageSize": 250}


def test_list_pages_omits_page_size_when_unset():
    store, paginator = _store_with_pages([{"vectors": [{"key": "k0"}]}])

    list(store.list_pages())

    kwargs = paginator.paginate.call_args.kwargs
    assert kwargs["PaginationConfig"] == {}
    assert kwargs["returnData"] is False
    assert kwargs["returnMetadata"] is False


@pytest.mark.parametrize("page_size", [0, -1, 1001])
def test_list_pages_rejects_out_of_range_page_size(page_size):
    store, _ = _store_with_pages([])

    with pytest.raises(ValueError, match="page_size must be between 1 and 1000"):
        list(store.list_pages(page_size=page_size))


def test_list_pages_follows_next_token_against_real_paginator():
    """Drive the real botocore paginator, so nextToken continuation and our
    request parameter names are checked against the live service model."""
    client = boto3.Session(
        aws_access_key_id="x", aws_secret_access_key="y", region_name="us-east-1"
    ).client("s3vectors")
    store = S3VectorsStore.__new__(S3VectorsStore)
    store._config = _config()
    store._client = client

    base = {
        "vectorBucketName": "test-bucket",
        "indexName": "test-index",
        "returnData": False,
        "returnMetadata": False,
    }
    with Stubber(client) as stub:
        stub.add_response(
            "list_vectors",
            {"vectors": [{"key": "k0"}, {"key": "k1"}], "nextToken": "page-2"},
            base,
        )
        stub.add_response(
            "list_vectors",
            {"vectors": [{"key": "k2"}]},  # no nextToken -> pagination stops
            {**base, "nextToken": "page-2"},
        )

        assert [v["key"] for p in store.list_pages() for v in p] == ["k0", "k1", "k2"]
        stub.assert_no_pending_responses()


# -------------------------------------------------------------- client: items
class FakeVectors:
    """Stands in for S3VectorsStore, recording how many pages were pulled."""

    def __init__(self, pages):
        self._pages = pages
        self.fetched = []
        self.calls = []

    def list_pages(self, return_data=False, return_metadata=False, page_size=None):
        self.calls.append(
            dict(return_data=return_data, return_metadata=return_metadata, page_size=page_size)
        )
        for page in self._pages:
            self.fetched.append(page)
            yield page


class FakeDocs:
    def __init__(self, store=None):
        self._store = store or {}  # (ns, id) -> {"text":..., "metadata":...}
        self.calls = []

    def get_many(self, namespace, ids):
        self.calls.append((namespace, list(ids)))
        return {i: self._store[(namespace, i)] for i in ids if (namespace, i) in self._store}


def _db(pages, docs=None):
    db = Dynavec.__new__(Dynavec)
    db._vectors = FakeVectors(pages)
    db._docs = docs or FakeDocs()
    return db


def _doc(text, **metadata):
    return {"text": text, "metadata": metadata}


def test_list_vectors_scopes_to_one_namespace():
    pages = [[{"key": "kb#a"}, {"key": "other#b"}, {"key": "kb#c"}]]
    docs = FakeDocs({("kb", "a"): _doc("A"), ("kb", "c"): _doc("C")})
    db = _db(pages, docs)

    got = list(db.list_vectors(namespace="kb"))

    assert [r.id for r in got] == ["a", "c"]
    assert [r.text for r in got] == ["A", "C"]
    # only the in-scope ids are ever hydrated
    assert docs.calls == [("kb", ["a", "c"])]


def test_list_vectors_walks_every_namespace_by_default():
    pages = [[{"key": "kb#a"}, {"key": "other#b"}]]
    docs = FakeDocs({("kb", "a"): _doc("A"), ("other", "b"): _doc("B")})
    db = _db(pages, docs)

    got = list(db.list_vectors())

    assert [(r.id, r.text) for r in got] == [("a", "A"), ("b", "B")]
    # hydration is grouped per namespace within the page
    assert sorted(docs.calls) == [("kb", ["a"]), ("other", ["b"])]


def test_list_vectors_stitches_pages_in_order():
    pages = [
        [{"key": "kb#a"}, {"key": "kb#b"}],
        [{"key": "kb#c"}],
        [{"key": "kb#d"}],
    ]
    db = _db(pages)

    assert [r.id for r in db.list_vectors(namespace="kb")] == ["a", "b", "c", "d"]


def test_list_vectors_empty_index_yields_nothing():
    db = _db([])

    assert list(db.list_vectors()) == []


def test_list_vectors_namespace_with_no_matches_yields_nothing():
    db = _db([[{"key": "other#b"}]])

    assert list(db.list_vectors(namespace="kb")) == []


def test_list_vectors_is_lazy():
    """The property that makes this a generator: only the first page is fetched
    before the first ``next()``, however many pages exist."""
    pages = [[{"key": f"kb#{i}"}] for i in range(5)]
    db = _db(pages)

    gen = db.list_vectors(namespace="kb")
    assert db._vectors.fetched == []  # nothing fetched before iteration starts

    first = next(gen)

    assert first.id == "0"
    assert db._vectors.fetched == pages[:1]

    next(gen)
    assert db._vectors.fetched == pages[:2]


def test_list_vectors_hydrates_text_and_metadata():
    docs = FakeDocs({("kb", "a"): _doc("hello", cat="food", extra=1)})
    db = _db([[{"key": "kb#a"}]], docs)

    (result,) = list(db.list_vectors(namespace="kb"))

    assert result.text == "hello"
    assert result.metadata == {"cat": "food", "extra": 1}
    assert result.score == 1.0
    assert result.distance is None
    assert result.vector is None
    # hydrating means we do not pay for metadata on the S3 Vectors side
    assert db._vectors.calls == [
        dict(return_data=False, return_metadata=False, page_size=None)
    ]


def test_list_vectors_without_hydration_uses_s3_metadata():
    page = [
        {
            "key": "kb#a",
            "metadata": {NS_METADATA_KEY: "kb", TEXT_METADATA_KEY: "mirrored", "cat": "food"},
        }
    ]
    docs = FakeDocs({("kb", "a"): _doc("from-dynamo")})
    db = _db([page], docs)

    (result,) = list(db.list_vectors(namespace="kb", hydrate=False))

    assert docs.calls == []  # DynamoDB is never touched
    assert result.text == "mirrored"
    assert result.metadata == {"cat": "food"}  # internal tags stripped
    assert db._vectors.calls == [
        dict(return_data=False, return_metadata=True, page_size=None)
    ]


def test_list_vectors_without_hydration_tolerates_missing_metadata():
    db = _db([[{"key": "kb#a"}]])

    (result,) = list(db.list_vectors(namespace="kb", hydrate=False))

    assert result.text is None
    assert result.metadata == {}


def test_list_vectors_include_vectors():
    page = [{"key": "kb#a", "data": {"float32": [0.1, 0.2, 0.3, 0.4]}}]
    db = _db([page])

    (result,) = list(db.list_vectors(namespace="kb", include_vectors=True))

    assert result.vector == [0.1, 0.2, 0.3, 0.4]
    assert db._vectors.calls[0]["return_data"] is True


def test_list_vectors_forwards_page_size():
    db = _db([[{"key": "kb#a"}]])

    list(db.list_vectors(namespace="kb", page_size=500))

    assert db._vectors.calls[0]["page_size"] == 500


def test_list_vectors_decodes_escaped_key_components():
    """Namespaces/ids containing the key separator round-trip through the scope
    check rather than being silently dropped."""
    db = _db([[{"key": "ten%23ant#doc%23one"}]])

    (result,) = list(db.list_vectors(namespace="ten#ant"))

    assert result.id == "doc#one"
