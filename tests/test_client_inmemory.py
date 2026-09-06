"""End-to-end orchestration test using in-memory fakes for both AWS stores.

Verifies the write/read wiring (embed -> split -> put; query -> hydrate -> rank)
without any AWS calls or boto3.
"""

import math

import pytest

import dynavec.client as client_mod
from dynavec import Document, Dynavec, DynavecConfig, SemanticCache
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

    def query_pages(self, query_vector, top_k, filter=None, return_metadata=True, return_distance=True):
        # emulate a paginator: split the result into pages of 2
        hits = self.query(query_vector, top_k, filter, return_metadata, return_distance)
        for i in range(0, len(hits), 2):
            yield hits[i : i + 2]

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

class WrongOutputEmbedder(HashEmbedder):
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


def test_embedder_output_dimension_mismatch_raises(db):
    from dynavec.exceptions import DimensionMismatchError

    db.embedder = WrongOutputEmbedder(8)

    with pytest.raises(
        DimensionMismatchError,
        match=r"Document 'x' vector has dimension 2, expected 8",
    ):
        db.upsert([Document(id="x", text="hello")])


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


def test_update_metadata_merge_preserves_vector(db):
    db.upsert([Document(id="1", text="apple pie", metadata={"cat": "food"})])
    before = db._vectors._store["default#1"][0]
    db.update("1", metadata={"rating": 5})
    after_meta = db.get(["1"])[0].metadata
    assert after_meta["cat"] == "food"  # preserved
    assert after_meta["rating"] == 5    # added
    # vector unchanged because neither text nor vector was updated
    assert db._vectors._store["default#1"][0] == before


def test_update_text_reembeds(db):
    db.upsert([Document(id="1", text="apple")])
    v_before = list(db._vectors._store["default#1"][0])
    db.update("1", text="rocket launch trajectory")
    v_after = list(db._vectors._store["default#1"][0])
    assert v_before != v_after
    assert db.get(["1"])[0].text == "rocket launch trajectory"


def test_update_missing_raises(db):
    from dynavec.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        db.update("nope", metadata={"x": 1})


def test_search_stream_yields_incrementally(db):
    db.upsert(
        [Document(id=str(i), text=f"apple item {i}") for i in range(5)]
    )
    gen = db.search_stream("apple", top_k=4)
    first = next(gen)
    assert first.text is not None
    rest = list(gen)
    assert 1 + len(rest) <= 4  # respects top_k cap


def test_namespace_view(db):
    kb = db.namespace("kb")
    kb.upsert([Document(id="1", text="hello world")])
    hits = kb.search("hello", top_k=3)
    assert hits[0].id == "1"
    assert kb.namespace == "kb"


def test_rescore_with_metric(db):
    db.upsert(
        [
            Document(id="1", text="apple pie recipe"),
            Document(id="2", text="apple orchard tour"),
            Document(id="3", text="rocket to mars"),
        ]
    )
    hits = db.search("apple", top_k=2, rescore={"cosine": 0.5, "manhattan": 0.5})
    assert len(hits) == 2


def test_search_can_normalize_final_scores(db):
    db.upsert(
        [
            Document(id="1", text="apple pie recipe"),
            Document(id="2", text="apple orchard tour"),
            Document(id="3", text="rocket to mars"),
        ]
    )

    db._cache = SemanticCache(threshold=0.999)
    db.search("apple", top_k=3, rescore="dot")

    hits = db.search("apple", top_k=3, rescore="dot", normalize_scores=True)

    assert db._cache.misses == 2
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert max(scores) == pytest.approx(1.0)
    assert min(scores) == pytest.approx(0.0)


def test_transform_applied_on_upsert(db):
    def tag(ctx):
        ctx.metadata["source"] = "unit-test"
        return ctx

    db.upsert([Document(id="1", text="hello")], transform=tag)
    assert db.get(["1"])[0].metadata["source"] == "unit-test"


def test_search_many_parallel(db):
    db.upsert([Document(id="1", text="apple"), Document(id="2", text="rocket")])
    results = db.search_many(["apple", "rocket"], top_k=1)
    assert len(results) == 2
    assert all(len(r) == 1 for r in results)


def test_context_manager_closes_pool(db):
    with db as d:
        d.upsert([Document(id="1", text="hi")])
    assert db._pool is None


def test_graph_search_scopes_to_related_docs(db):
    db.upsert(
        [
            Document(id="d1", text="apple pie recipe"),
            Document(id="d2", text="apple orchard tour"),
            Document(id="d3", text="rocket launch"),  # unrelated, not linked
        ]
    )
    # entities: fruit -> [d1, d2]; graph scopes search to those docs
    db.graph_add_node("fruit", ntype="topic")
    db.graph_link("fruit", ["d1", "d2"])

    hits = db.graph_search("apple", seed_entities=["fruit"], top_k=5)
    assert {h.id for h in hits} == {"d1", "d2"}  # d3 excluded by the graph


def test_graph_traversal_hops(db):
    db.upsert([Document(id="d1", text="x"), Document(id="d2", text="y")])
    db.graph_add_edge("a", "related_to", "b")
    db.graph_link("b", ["d1"])
    # 1 hop from 'a' reaches 'b'
    assert db.graph_neighbors("a", hops=1) == ["b"]
    hits = db.graph_search("x", seed_entities=["a"], hops=1, top_k=5)
    assert {h.id for h in hits} == {"d1"}


def test_semantic_cache_hits_on_repeat(db):
    from dynavec.cache import SemanticCache

    db._cache = SemanticCache(threshold=0.99)
    db.upsert([Document(id="1", text="apple pie")])

    # count store queries to prove the 2nd call is served from cache
    calls = {"n": 0}
    real_query = db._vectors.query

    def counting_query(*a, **k):
        calls["n"] += 1
        return real_query(*a, **k)

    db._vectors.query = counting_query

    assert db.cache is db._cache
    r1 = db.search("apple pie", top_k=3)
    assert db.cache.stats() == {"hits": 0, "misses": 1, "hit_rate": 0.0}
    r2 = db.search("apple pie", top_k=3)  # identical -> cache hit
    assert calls["n"] == 1
    assert [x.id for x in r1] == [x.id for x in r2]
    assert db.cache.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}


def test_ingest_chunks_and_stores(db):
    from dynavec.ingest import IterableSource, ingest

    source = IterableSource(
        [
            {"id": "doc1", "text": "a" * 25, "metadata": {"src": "wiki"}},
            {"id": "doc2", "text": "b" * 10},
        ]
    )
    n = ingest(db, source, chunk_size=10, overlap=0, batch_size=4)
    assert n == 3  # doc1 -> 2 unique chunks, doc2 -> 1 chunk
    got = db.get(["doc1#chunk0"])[0]
    assert got.metadata["source_id"] == "doc1"
    assert got.metadata["src"] == "wiki"
