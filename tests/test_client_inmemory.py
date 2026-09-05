"""End-to-end orchestration test using in-memory fakes for both AWS stores.

Verifies the write/read wiring (embed -> split -> put; query -> hydrate -> rank)
without any AWS calls or boto3.
"""

import math

import pytest

import dynavec.client as client_mod
from dynavec import Document, Dynavec, DynavecConfig
from dynavec.config import NS_METADATA_KEY
from dynavec.embeddings.base import Embedder


class HashEmbedder(Embedder):
    """Deterministic tiny embedder: maps text to a fixed-dim vector."""

    def __init__(self, dimension=8):
        self.dimension = dimension

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dimension
            for i, ch in enumerate(t):
                v[i % self.dimension] += (ord(ch) % 17) / 17.0
            out.append(v)
        return out


def _cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return 1.0 - dot / (na * nb)


class FakeS3(client_mod.S3VectorsStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._store = {}  # key -> (vector, metadata)

    def put_vectors(self, vectors):
        for key, vec, meta in vectors:
            self._store[key] = (list(vec), dict(meta))

    def _matches(self, meta, filt):
        if not filt:
            return True
        clauses = filt["$and"] if "$and" in filt else [filt]
        for clause in clauses:
            for k, v in clause.items():
                if meta.get(k) != v:
                    return False
        return True

    def query(self, query_vector, top_k, filter=None, return_metadata=True, return_distance=True):
        scored = []
        for key, (vec, meta) in self._store.items():
            if not self._matches(meta, filter):
                continue
            scored.append((key, _cosine_distance(query_vector, vec), meta))
        scored.sort(key=lambda x: x[1])
        return [{"key": k, "distance": d, "metadata": m} for k, d, m in scored[:top_k]]

    def get_vectors(self, keys, return_metadata=False):
        return {k: {"vector": self._store[k][0], "metadata": self._store[k][1]} for k in keys if k in self._store}

    def delete_vectors(self, keys):
        for k in keys:
            self._store.pop(k, None)


class FakeDDB(client_mod.DynamoDBStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._store = {}  # (ns, id) -> {text, metadata}

    def put_many(self, namespace, items):
        for doc_id, text, meta in items:
            self._store[(namespace, doc_id)] = {"text": text, "metadata": dict(meta)}

    def get_many(self, namespace, ids):
        return {i: self._store[(namespace, i)] for i in ids if (namespace, i) in self._store}

    def delete_many(self, namespace, ids):
        for i in ids:
            self._store.pop((namespace, i), None)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=8)
    return Dynavec(cfg, embedder=HashEmbedder(8))


def test_upsert_and_search_roundtrip(db):
    res = db.upsert(
        [
            Document(id="1", text="apple pie recipe", metadata={"cat": "food"}),
            Document(id="2", text="rocket launch schedule", metadata={"cat": "space"}),
            Document(id="3", text="apple orchard tour", metadata={"cat": "food"}),
        ]
    )
    assert res.count == 3

    hits = db.search("apple", top_k=2)
    assert len(hits) == 2
    assert hits[0].text is not None
    # scores descending, higher = more similar
    assert hits[0].score >= hits[1].score


def test_metadata_filter_scopes_results(db):
    db.upsert(
        [
            Document(id="1", text="apple pie", metadata={"cat": "food"}),
            Document(id="2", text="apple satellite", metadata={"cat": "space"}),
        ]
    )
    hits = db.search("apple", top_k=5, filter={"cat": "space"})
    assert {h.id for h in hits} == {"2"}


def test_namespace_isolation(db):
    db.upsert([Document(id="1", text="hello")], namespace="tenantA")
    db.upsert([Document(id="1", text="world")], namespace="tenantB")
    a = db.search("hello", top_k=5, namespace="tenantA")
    b = db.search("hello", top_k=5, namespace="tenantB")
    assert a[0].text == "hello"
    assert b[0].text == "world"


def test_dimension_mismatch_raises(db):
    from dynavec.exceptions import DimensionMismatchError

    with pytest.raises(DimensionMismatchError):
        db.upsert([Document(id="x", vector=[0.1, 0.2])])  # wrong dim (2 != 8)


def test_auto_metadata_switch(db):
    db.upsert([Document(id="1", text="hello world")], auto_metadata=True)
    got = db.get(["1"])[0]
    assert "content_hash" in got.metadata
    assert got.metadata["word_count"] == 2


def test_delete(db):
    db.upsert([Document(id="1", text="hello")])
    db.delete(["1"])
    assert db.get(["1"]) == []


def test_ns_tag_present_in_s3(db):
    db.upsert([Document(id="1", text="hello")], namespace="ns9")
    # reach into the fake to confirm the namespace tag was written
    store = db._vectors._store
    (_, meta), = [v for k, v in store.items()]
    assert meta[NS_METADATA_KEY] == "ns9"
