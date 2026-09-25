"""Unit tests for NamespaceView.search_many()."""

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

    def query_pages(
        self,
        query_vector,
        top_k,
        filter=None,
        return_metadata=True,
        return_distance=True,
        page_size=None,
    ):
        # emulate a paginator: split the result into pages of 2
        hits = self.query(query_vector, top_k, filter, return_metadata, return_distance)
        chunk_size = page_size or 2
        for i in range(0, len(hits), chunk_size):
            yield hits[i : i + chunk_size]

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
class FakeGraph(client_mod.GraphStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._nodes = {}  # (ns, entity_id) -> {"edges": [...], "docs": [...]}

    def _node(self, ns, eid):
        return self._nodes.setdefault((ns, eid), {"edges": [], "docs": []})

    def add_node(self, ns, entity_id, ntype=None, props=None):
        self._node(ns, entity_id)

    def add_edge(self, ns, src, relation, dst):
        self._node(ns, src)["edges"].append({"relation": relation, "target": dst})
        self._node(ns, dst)

    def link_docs(self, ns, entity_id, doc_ids):
        self._node(ns, entity_id)["docs"].extend(doc_ids)

    def delete_edge(self, ns, src, relation, dst):
        node = self.get_node(ns, src)
        if not node:
            return 0
        before = len(node["edges"])
        node["edges"] = [
            e for e in node["edges"] if not (e["relation"] == relation and e["target"] == dst)
        ]
        return before - len(node["edges"])

    def delete_node(self, ns, entity_id):
        removed = 0
        for (node_ns, eid), node in self._nodes.items():
            if node_ns != ns or eid == entity_id:
                continue
            before = len(node["edges"])
            node["edges"] = [e for e in node["edges"] if e["target"] != entity_id]
            removed += before - len(node["edges"])
        self._nodes.pop((ns, entity_id), None)
        return removed

    def get_node(self, ns, entity_id):
        return self._nodes.get((ns, entity_id))

    def neighbors(self, ns, entity_id, relation=None):
        node = self.get_node(ns, entity_id)
        if not node:
            return []
        return [
            e["target"] for e in node["edges"]
            if relation is None or e["relation"] == relation
        ]

    def get_docs(self, ns, entity_ids):
        seen, out = set(), []
        for eid in entity_ids:
            node = self.get_node(ns, eid)
            if not node:
                continue
            for d in node["docs"]:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
        return out
@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=8)
    return Dynavec(cfg, embedder=HashEmbedder(8))
def test_search_many_forwards_multiple_queries_with_namespace_isolation(db):
    kb = db.namespace("kb")
    other = db.namespace("other")
    kb.upsert([Document(id="1", text="apple"), Document(id="2", text="rocket")])
    other.upsert([Document(id="1", text="apple"), Document(id="2", text="rocket")])

    results = kb.search_many(["apple", "rocket"], top_k=1)

    assert len(results) == 2
    assert all(len(r) == 1 for r in results)
    assert all(r[0].id in ("1", "2") for r in results)


def test_search_many_forwards_kwargs(db):
    kb = db.namespace("kb")
    kb.upsert(
        [
            Document(id="1", text="apple pie", metadata={"kind": "fruit"}),
            Document(id="2", text="rocket launch", metadata={"kind": "space"}),
        ]
    )

    results = kb.search_many(["apple", "rocket"], top_k=5, filter={"kind": "fruit"})

    assert all(len(r) <= 1 for r in results)


def test_search_many_empty_batch(db):
    kb = db.namespace("kb")
    kb.upsert([Document(id="1", text="apple")])

    assert kb.search_many([]) == []