"""Tests for namespace export and import (JSONL dump and restore)."""

import io
import json
import math

import pytest

import dynavec.client as client_mod
from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings.base import Embedder


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

    def put_many(self, namespace, items):
        for doc_id, text, meta in items:
            self._store[(namespace, doc_id)] = {"text": text, "metadata": dict(meta)}

    def get_many(self, namespace, ids):
        return {i: self._store[(namespace, i)] for i in ids if (namespace, i) in self._store}

    def delete_many(self, namespace, ids):
        for i in ids:
            self._store.pop((namespace, i), None)


def _make(monkeypatch, **cfg_kw):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=8, **cfg_kw)
    return Dynavec(cfg, embedder=HashEmbedder(8))


def test_export_namespace_to_file(monkeypatch, tmp_path):
    db = _make(monkeypatch)
    db.upsert(
        [
            Document(id="doc1", text="first document", metadata={"category": "tech"}),
            Document(id="doc2", text="second document", metadata={"category": "science"}),
        ],
        namespace="tenant-a",
    )
    db.upsert(
        [Document(id="other", text="other tenant doc")],
        namespace="tenant-b",
    )

    out_file = tmp_path / "tenant_a.jsonl"
    count = db.export_namespace(out_file, namespace="tenant-a")
    assert count == 2

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    by_id = {item["id"]: item for item in lines}
    assert "doc1" in by_id
    assert "doc2" in by_id
    assert "other" not in by_id
    assert by_id["doc1"]["text"] == "first document"
    assert by_id["doc1"]["metadata"] == {"category": "tech"}
    assert len(by_id["doc1"]["vector"]) == 8


def test_export_namespace_to_stream(monkeypatch):
    db = _make(monkeypatch)
    db.upsert([Document(id="doc1", text="hello")], namespace="default")

    stream = io.StringIO()
    count = db.export_namespace(stream, namespace="default")
    assert count == 1

    content = stream.getvalue().strip()
    data = json.loads(content)
    assert data["id"] == "doc1"
    assert data["text"] == "hello"


def test_import_namespace_from_file(monkeypatch, tmp_path):
    db = _make(monkeypatch)
    in_file = tmp_path / "import.jsonl"
    in_file.write_text(
        json.dumps({"id": "imp1", "vector": [0.1] * 8, "text": "imported text", "metadata": {"x": 1}}) + "\n"
        + json.dumps({"id": "imp2", "vector": [0.2] * 8, "text": "second imported", "metadata": {"x": 2}}) + "\n",
        encoding="utf-8",
    )

    count = db.import_namespace(in_file, namespace="restored-ns", batch_size=1)
    assert count == 2

    # Verify documents are searchable in the restored namespace
    hits = db.search("imported", top_k=2, namespace="restored-ns")
    assert len(hits) == 2
    found_ids = {h.id for h in hits}
    assert found_ids == {"imp1", "imp2"}


def test_import_namespace_from_stream(monkeypatch):
    db = _make(monkeypatch)
    jsonl_stream = io.StringIO(
        json.dumps({"id": "s1", "vector": [0.5] * 8, "text": "streamed doc", "metadata": {}}) + "\n"
    )

    count = db.import_namespace(jsonl_stream, namespace="stream-ns")
    assert count == 1

    hits = db.get(["s1"], namespace="stream-ns")
    assert len(hits) == 1
    assert hits[0].text == "streamed doc"


def test_roundtrip_export_and_import(monkeypatch, tmp_path):
    db = _make(monkeypatch)
    orig_docs = [
        Document(id="1", text="apple pie", metadata={"tag": "fruit"}),
        Document(id="2", text="banana split", metadata={"tag": "fruit"}),
        Document(id="3", text="carrot cake", metadata={"tag": "veggie"}),
    ]
    db.upsert(orig_docs, namespace="source")

    dump_path = tmp_path / "dump.jsonl"
    assert db.export_namespace(dump_path, namespace="source") == 3

    assert db.import_namespace(dump_path, namespace="target") == 3

    target_docs = list(db.iter_namespace("target"))
    assert len(target_docs) == 3
    orig_by_id = {d.id: d for d in orig_docs}
    for td in target_docs:
        od = orig_by_id[td["id"]]
        assert td["text"] == od.text
        assert td["metadata"] == od.metadata
        assert len(td["vector"]) == 8


def test_namespace_view_export_and_import(monkeypatch, tmp_path):
    db = _make(monkeypatch)
    view = db.namespace("view-tenant")
    view.upsert([Document(id="v1", text="view doc", metadata={"view": True})])

    file_path = tmp_path / "view.jsonl"
    assert view.export_namespace(file_path) == 1

    # Restore into another view
    view2 = db.namespace("view-tenant-2")
    assert view2.import_namespace(file_path) == 1

    items = list(view2)
    assert len(items) == 1
    assert items[0]["id"] == "v1"
    assert items[0]["text"] == "view doc"


def test_import_validation_errors(monkeypatch):
    db = _make(monkeypatch)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        db.import_namespace(io.StringIO(), batch_size=0)

    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        db.import_namespace(io.StringIO("not-valid-json\n"))

    with pytest.raises(ValueError, match="Missing required 'id' or 'vector' field at line 1"):
        db.import_namespace(io.StringIO('{"text": "no id or vector"}\n'))
