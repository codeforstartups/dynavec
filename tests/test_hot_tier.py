"""In-memory hot tier: correctness + that warmed namespaces skip S3/DynamoDB.

Reuses the in-memory fakes so no AWS is touched. Where it matters we assert on
call counts to prove the hot path really bypasses S3 Vectors and DynamoDB.
"""

import math

import pytest

import dynavec.client as client_mod
from dynavec import Document, Dynavec, DynavecConfig
from dynavec.config import NS_METADATA_KEY, TEXT_METADATA_KEY
from dynavec.embeddings.base import Embedder
from dynavec.hot import UnsupportedFilter, matches


class HashEmbedder(Embedder):
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
        self.query_calls = 0

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
        self.query_calls += 1
        scored = []
        for key, (vec, meta) in self._store.items():
            if not self._matches(meta, filter):
                continue
            scored.append((key, _cosine_distance(query_vector, vec), meta))
        scored.sort(key=lambda x: x[1])
        return [{"key": k, "distance": d, "metadata": m} for k, d, m in scored[:top_k]]

    def get_vectors(self, keys, return_metadata=False):
        return {
            k: {"vector": self._store[k][0], "metadata": self._store[k][1]}
            for k in keys if k in self._store
        }

    def list_pages(self, return_data=False, return_metadata=False, page_size=None):
        page = []
        for key, (vec, meta) in self._store.items():
            entry = {"key": key}
            if return_data:
                entry["data"] = {"float32": list(vec)}
            if return_metadata:
                entry["metadata"] = dict(meta)
            page.append(entry)
        if page:
            yield page

    def delete_vectors(self, keys):
        for k in keys:
            self._store.pop(k, None)


class FakeDDB(client_mod.DynamoDBStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._store = {}
        self.get_calls = 0

    def put_many(self, namespace, items):
        for doc_id, text, meta in items:
            self._store[(namespace, doc_id)] = {"text": text, "metadata": dict(meta)}

    def get_many(self, namespace, ids):
        self.get_calls += 1
        return {i: self._store[(namespace, i)] for i in ids if (namespace, i) in self._store}

    def delete_many(self, namespace, ids):
        for i in ids:
            self._store.pop((namespace, i), None)


def _make(monkeypatch, **cfg_kw):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=8, **cfg_kw)
    return Dynavec(cfg, embedder=HashEmbedder(8))


# --------------------------------------------------------------------- matcher

def test_matches_supported_operators():
    m = {"cat": "food", "year": 2026, "score": 0.4}
    assert matches(m, {"cat": "food"})
    assert matches(m, {"year": {"$gte": 2020}})
    assert not matches(m, {"year": {"$lt": 2020}})
    assert matches(m, {"cat": {"$in": ["food", "space"]}})
    assert matches(m, {"$and": [{"cat": "food"}, {"year": {"$gte": 2026}}]})
    assert matches(m, {"$or": [{"cat": "space"}, {"year": 2026}]})
    assert not matches(m, {"cat": "space"})


def test_matches_rejects_unsupported_operator():
    with pytest.raises(UnsupportedFilter):
        matches({"a": 1}, {"a": {"$regex": ".*"}})


# ---------------------------------------------------------------- integration

def test_disabled_by_default_uses_s3(monkeypatch):
    db = _make(monkeypatch)
    assert db._hot is None
    db.upsert([Document(id="1", text="apple pie")])
    hits = db.search("apple", top_k=1)
    assert hits and db._vectors.query_calls == 1  # S3 was queried


def test_warm_serves_from_ram_without_touching_s3_or_ddb(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert(
        [
            Document(id="1", text="apple pie recipe", metadata={"cat": "food"}),
            Document(id="2", text="rocket launch", metadata={"cat": "space"}),
            Document(id="3", text="apple orchard", metadata={"cat": "food"}),
        ]
    )
    loaded = db.warm()
    assert loaded == 3
    assert db._hot.is_authoritative("default")

    s3_before = db._vectors.query_calls
    ddb_before = db._docs.get_calls
    hits = db.search("apple", top_k=2)

    assert len(hits) == 2
    assert hits[0].text is not None                 # text served from RAM
    assert hits[0].score >= hits[1].score
    assert db._vectors.query_calls == s3_before     # no S3 Vectors query
    assert db._docs.get_calls == ddb_before         # no DynamoDB hydration


def test_write_through_keeps_warmed_namespace_current(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert([Document(id="1", text="apple pie")])
    db.warm()
    s3_before = db._vectors.query_calls
    # New doc after warming must be found via the hot path (write-through).
    db.upsert([Document(id="2", text="apple tart")])
    hits = db.search("apple tart", top_k=2)
    assert {h.id for h in hits} == {"1", "2"}
    assert db._vectors.query_calls == s3_before      # still served from RAM


def test_filter_on_hot_path(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert(
        [
            Document(id="1", text="apple pie", metadata={"cat": "food"}),
            Document(id="2", text="apple satellite", metadata={"cat": "space"}),
        ]
    )
    db.warm()
    s3_before = db._vectors.query_calls
    hits = db.search("apple", top_k=5, filter={"cat": "space"})
    assert {h.id for h in hits} == {"2"}
    assert db._vectors.query_calls == s3_before      # filtered in RAM


def test_unsupported_filter_falls_back_to_s3(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert([Document(id="1", text="apple", metadata={"n": 3})])
    db.warm()
    assert db._hot.is_authoritative("default")
    # The hot path can't honor $regex, so it must return None (not a wrong/empty
    # result) and let the client fall back to the S3 Vectors path.
    assert db._hot.search("default", [0.0] * 8, 5, {"n": {"$regex": "3"}}) is None
    s3_before = db._vectors.query_calls
    db.search("apple", top_k=5, filter={"n": {"$regex": "3"}})
    assert db._vectors.query_calls == s3_before + 1   # fell back to S3


def test_delete_removes_from_hot_tier(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert([Document(id="1", text="apple"), Document(id="2", text="apple two")])
    db.warm()
    db.delete(["1"])
    s3_before = db._vectors.query_calls
    hits = db.search("apple", top_k=5)
    assert {h.id for h in hits} == {"2"}
    assert db._vectors.query_calls == s3_before      # still hot, id 1 gone


def test_capacity_cap_falls_back_to_s3(monkeypatch):
    db = _make(monkeypatch, hot_tier=True, hot_tier_max_vectors=2)
    db.upsert([Document(id=str(i), text=f"doc {i}") for i in range(5)])
    loaded = db.warm()  # 5 > cap of 2
    assert loaded == 0
    assert not db._hot.is_authoritative("default")
    hits = db.search("doc", top_k=3)
    assert db._vectors.query_calls >= 1              # fell back to S3
    assert hits


def test_warm_hydrates_text_from_ddb(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    db.upsert([Document(id="1", text="the canonical text", metadata={"k": "v"})])
    db.warm()
    hits = db.search("canonical", top_k=1)
    assert hits[0].text == "the canonical text"
    assert hits[0].metadata.get("k") == "v"
    assert NS_METADATA_KEY not in hits[0].metadata
    assert TEXT_METADATA_KEY not in hits[0].metadata


def test_warm_requires_hot_tier_enabled(monkeypatch):
    from dynavec.exceptions import ConfigurationError

    db = _make(monkeypatch)  # hot_tier defaults off
    with pytest.raises(ConfigurationError):
        db.warm()


def test_hot_stats(monkeypatch):
    db = _make(monkeypatch, hot_tier=True)
    assert db.hot_stats() == {
        "authoritative_namespaces": [],
        "resident_vectors": 0,
        "max_vectors": 200_000,
        "namespaces": {},
    }
    db.upsert([Document(id="1", text="apple")])
    db.warm()
    stats = db.hot_stats()
    assert stats["authoritative_namespaces"] == ["default"]
    assert stats["resident_vectors"] == 1
