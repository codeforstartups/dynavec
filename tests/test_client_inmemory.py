"""End-to-end orchestration test using in-memory fakes for both AWS stores.

Verifies the write/read wiring (embed -> split -> put; query -> hydrate -> rank)
without any AWS calls or boto3.
"""

import math
import sys
from types import SimpleNamespace

import pytest

import dynavec.client as client_mod
from dynavec import (
    Document,
    Dynavec,
    DynavecConfig,
    ExplainedSearchResult,
    SemanticCache,
)
from dynavec.config import NS_METADATA_KEY
from dynavec.embeddings.base import Embedder
from dynavec.exceptions import ConfigurationError


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
        self._store = {}  # (ns, id) -> {text, metadata[, version]}

    def put_many(self, namespace, items):
        # like a real put_item: replaces the whole item, dropping any version
        for doc_id, text, meta in items:
            self._store[(namespace, doc_id)] = {"text": text, "metadata": dict(meta)}

    def put_versioned(self, namespace, doc_id, text, metadata, expected_version):
        from dynavec.exceptions import ConflictError

        stored = self._store.get((namespace, doc_id), {})
        if stored.get("version", 0) != expected_version:
            raise ConflictError(doc_id, namespace, expected_version)
        self._store[(namespace, doc_id)] = {
            "text": text,
            "metadata": dict(metadata),
            "version": expected_version + 1,
        }
        return expected_version + 1

    def get_versioned(self, namespace, doc_id):
        stored = self._store.get((namespace, doc_id))
        if stored is None:
            return None
        return {**stored, "version": stored.get("version", 0)}

    def get_many(self, namespace, ids):
        return {
            i: {"text": self._store[(namespace, i)]["text"],
                "metadata": self._store[(namespace, i)]["metadata"]}
            for i in ids if (namespace, i) in self._store
        }

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

    cfg = DynavecConfig(
        vector_bucket="b",
        index="i",
        table="t",
        dimension=8,
    )
    return Dynavec(cfg, embedder=HashEmbedder(8))

@pytest.fixture
def cross_encoder_db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)

    cfg = DynavecConfig(
        vector_bucket="b",
        index="i",
        table="t",
        dimension=8,
        cross_encoder_model="toy-cross-encoder",
    )
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


def test_search_without_explain_returns_results_list(db):
    db.upsert([Document(id="1", text="apple pie")])

    result = db.search("apple")

    assert isinstance(result, list)
    assert len(result) == 1


def test_graph_shortest_path_respects_hop_cap(db):
    db.graph_add_edge("a", "related_to", "b")
    db.graph_add_edge("b", "related_to", "c")
    db.graph_add_edge("c", "related_to", "d")

    assert db.graph_shortest_path("a", "d", hops=2) == []

def test_graph_shortest_path_returns_minimum_hops(db):
    db.graph_add_edge("a", "related_to", "c")
    db.graph_add_edge("c", "related_to", "e")
    db.graph_add_edge("e", "related_to", "d")

    db.graph_add_edge("a", "related_to", "b")
    db.graph_add_edge("b", "related_to", "d")

    assert db.graph_shortest_path("a", "d", hops=3) == [
        "a",
        "b",
        "d",
    ]

def test_search_explain_handles_empty_results(db):
    result = db.search("missing", explain=True)

    assert isinstance(result, ExplainedSearchResult)
    assert result.results == []

    explanation = result.explanation

    assert explanation.candidate_counts["retrieved"] == 0
    assert explanation.candidate_counts["final"] == 0
    assert explanation.timings_ms["vector_search"] >= 0
    assert explanation.timings_ms["total"] >= 0


def test_search_explain_reports_cache_hit(db):
    db._cache = SemanticCache(threshold=0.99)
    db.upsert([Document(id="1", text="apple pie")])

    db.search("apple pie", top_k=3)

    result = db.search("apple pie", top_k=3, explain=True)

    assert isinstance(result, ExplainedSearchResult)
    assert len(result.results) == 1

    explanation = result.explanation

    assert explanation.candidate_counts["cached"] == 1
    assert explanation.candidate_counts["final"] == 1
    assert explanation.timings_ms["cache_lookup"] >= 0
    assert explanation.timings_ms["total"] >= 0

    assert "vector_search" not in explanation.timings_ms
    assert "hydration" not in explanation.timings_ms


def test_search_explain_records_rescore_stage(db):
    db.upsert(
        [
            Document(id="1", text="apple pie recipe"),
            Document(id="2", text="apple orchard tour"),
            Document(id="3", text="rocket launch"),
        ]
    )

    result = db.search(
        "apple",
        top_k=2,
        rescore="cosine",
        explain=True,
    )

    assert isinstance(result, ExplainedSearchResult)
    assert len(result.results) == 2

    explanation = result.explanation

    assert explanation.timings_ms["rescore"] >= 0
    assert explanation.candidate_counts["rescored"] >= 2
    assert explanation.candidate_counts["final"] == 2


def test_search_explain_returns_structured_debug_result(db):
    db.upsert(
        [
            Document(id="1", text="apple pie recipe"),
            Document(id="2", text="rocket launch schedule"),
            Document(id="3", text="apple orchard tour"),
        ]
    )

    result = db.search("apple", top_k=2, explain=True)

    assert isinstance(result, ExplainedSearchResult)
    assert len(result.results) == 2

    explanation = result.explanation

    assert explanation.timings_ms["query_vector"] >= 0
    assert explanation.timings_ms["vector_search"] >= 0
    assert explanation.timings_ms["hydration"] >= 0
    assert explanation.timings_ms["total"] >= 0

    assert explanation.candidate_counts["retrieved"] >= 2
    assert explanation.candidate_counts["hydrated"] >= 2
    assert explanation.candidate_counts["final"] == 2


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
    (_, meta), = (v for k, v in store.items())
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


# ------------------------------------------------------ optimistic concurrency
def test_update_bumps_version_on_each_write(db):
    db.upsert([Document(id="1", text="apple pie")])
    assert db.update("1", metadata={"a": 1}).version == 1
    assert db.update("1", metadata={"b": 2}).version == 2


def test_update_with_stale_expected_version_writes_nothing(db):
    from dynavec.exceptions import ConflictError

    db.upsert([Document(id="1", text="apple pie", metadata={"cat": "food"})])
    first = db.update("1", metadata={"rating": 4})
    db.update("1", metadata={"rating": 5})  # someone else moves it to version 2
    vector_before = list(db._vectors._store["default#1"][0])

    with pytest.raises(ConflictError) as info:
        db.update("1", text="rocket launch", expected_version=first.version)

    assert info.value.expected_version == 1
    assert db.get(["1"])[0].text == "apple pie"
    assert db.get(["1"])[0].metadata["rating"] == 5
    assert db._vectors._store["default#1"][0] == vector_before


def test_concurrent_update_between_read_and_write_is_not_lost(db):
    from dynavec.exceptions import ConflictError

    db.upsert([Document(id="1", text="apple pie", metadata={"cat": "food"})])
    db.update("1", metadata={"views": 1})
    real_read = db._docs.get_versioned

    def read_then_race(namespace, doc_id):
        snapshot = real_read(namespace, doc_id)
        # another writer commits after our read but before our write
        db._docs.put_versioned(
            namespace, doc_id, snapshot["text"], {**snapshot["metadata"], "views": 2},
            expected_version=snapshot["version"],
        )
        return snapshot

    db._docs.get_versioned = read_then_race

    with pytest.raises(ConflictError):
        db.update("1", metadata={"tag": "dessert"})

    meta = db.get(["1"])[0].metadata
    assert meta["views"] == 2  # the concurrent write survived
    assert "tag" not in meta


def test_upsert_invalidates_earlier_versions(db):
    from dynavec.exceptions import ConflictError

    db.upsert([Document(id="1", text="apple pie")])
    v1 = db.update("1", metadata={"a": 1}).version
    db.upsert([Document(id="1", text="apple tart")])  # blind overwrite, unversioned

    with pytest.raises(ConflictError):
        db.update("1", metadata={"b": 2}, expected_version=v1)
    assert db.update("1", metadata={"b": 2}).version == 1


def test_update_upsert_if_missing_starts_at_version_one(db):
    res = db.update("new", text="fresh doc", upsert_if_missing=True)
    assert res.version == 1
    assert db.get(["new"])[0].text == "fresh doc"


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

def test_hybrid_graph_search_fuses_ann_and_graph_results(db):
    db.upsert(
        [
            Document(id="d1", text="apple pie recipe"),
            Document(id="d2", text="apple orchard tour"),
            Document(id="d3", text="rocket launch"),
        ]
    )

    db.graph_add_node("fruit", ntype="topic")
    db.graph_link("fruit", ["d1", "d2"])

    hits = db.hybrid_graph_search(
        "apple",
        seed_entities=["fruit"],
        top_k=3,
    )

    assert {hit.id for hit in hits} == {"d1", "d2", "d3"}


def test_graph_traversal_hops(db):
    db.upsert([Document(id="d1", text="x"), Document(id="d2", text="y")])
    db.graph_add_edge("a", "related_to", "b")
    db.graph_link("b", ["d1"])
    # 1 hop from 'a' reaches 'b'
    assert db.graph_neighbors("a", hops=1) == ["b"]
    hits = db.graph_search("x", seed_entities=["a"], hops=1, top_k=5)
    assert {h.id for h in hits} == {"d1"}


def test_graph_traversal_handles_cycles(db):
    db.graph_add_edge("a", "related_to", "b")
    db.graph_add_edge("b", "related_to", "c")
    db.graph_add_edge("c", "related_to", "a")

    assert set(db.graph_neighbors("a", hops=10)) == {"b", "c"}


def test_graph_delete_edge_bidirectional(db):
    db.graph_add_edge("a", "related_to", "b", bidirectional=True)

    assert db.graph_delete_edge("a", "related_to", "b", bidirectional=True) == 2
    assert db.graph_neighbors("a") == []
    assert db.graph_neighbors("b") == []
    assert db.graph_delete_edge("a", "related_to", "b", bidirectional=True) == 0


def test_graph_delete_node_drops_it_from_traversal(db):
    db.graph_add_edge("a", "related_to", "b")
    db.graph_add_edge("b", "related_to", "c")

    assert db.graph_delete_node("b") == 1
    assert db.graph_neighbors("a", hops=10) == []
    assert db.graph_delete_node("b") == 0


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


def test_reserved_key_separator_is_escaped_before_writes(db):
    db.upsert(
        [Document(id="doc#one", vector=[1.0] * 8)],
        namespace="tenant#one",
    )

    assert list(db._vectors._store) == ["tenant%23one#doc%23one"]
    assert db._docs._store[("tenant#one", "doc#one")]["text"] is None


def test_oversized_document_fails_upsert_before_any_write(db):
    from dynavec import ItemTooLargeError

    with pytest.raises(ItemTooLargeError) as info:
        db.upsert(
            [
                Document(id="small", text="fits"),
                Document(id="huge", text="x" * 500_000),
            ]
        )

    assert info.value.doc_id == "huge"
    assert info.value.size_bytes > info.value.limit_bytes
    assert "'huge'" in str(info.value) and "chunk_text" in str(info.value)
    # neither store saw the batch, so S3 Vectors and DynamoDB stay in sync
    assert db._vectors._store == {}
    assert db._docs._store == {}


def test_oversized_metadata_fails_update_and_keeps_the_stored_document(db):
    from dynavec import ItemTooLargeError

    db.upsert([Document(id="1", text="apple pie", metadata={"cat": "food"})])

    with pytest.raises(ItemTooLargeError):
        db.update("1", metadata={"blob": "y" * 500_000})

    assert db._docs._store[("default", "1")]["metadata"] == {"cat": "food"}


def test_search_records_telemetry(db):
    from dynavec.telemetry import TelemetryRecorder

    rec = TelemetryRecorder()
    db._telemetry = rec
    db.upsert([Document(id="1", text="apple pie"), Document(id="2", text="rocket")])
    db.search("apple", top_k=2, namespace="default")
    evs = rec.events()
    assert len(evs) == 1
    e = evs[0]
    assert e.op == "search"
    assert e.namespace == "default"
    assert e.n_results >= 1
    assert e.latency_ms >= 0
    assert e.status == "ok"
    assert e.cache_hit is None  # no cache configured


def test_search_telemetry_marks_cache_hit(db):
    from dynavec.cache import SemanticCache
    from dynavec.telemetry import TelemetryRecorder

    rec = TelemetryRecorder()
    db._telemetry = rec
    db._cache = SemanticCache(threshold=0.99)
    db.upsert([Document(id="1", text="apple pie")])
    db.search("apple pie", top_k=3)   # miss -> populates cache
    db.search("apple pie", top_k=3)   # hit
    hits = [e.cache_hit for e in rec.events()]
    assert True in hits and False in hits

def test_cross_encoder_rerank_on_toy_data(monkeypatch, cross_encoder_db):
    class FakeCrossEncoder:
        instance = None

        def __init__(self, model_name):
            self.model_name = model_name
            FakeCrossEncoder.instance = self

        def predict(self, pairs):
            return [
                0.95 if "target document" in document else 0.10
                for _, document in pairs
            ]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    cross_encoder_db.upsert([
        Document(
            id="d1",
            text="ordinary document",
            vector=[0.1] * 8,
        ),
        Document(
            id="d2",
            text="target document",
            vector=[0.1] * 8,
        ),
    ])

    results = cross_encoder_db.search(
        "find the target",
        top_k=1,
        rerank="cross-encoder",
    )

    assert results[0].id == "d2"
    assert results[0].score == 0.95
    assert FakeCrossEncoder.instance.model_name == "toy-cross-encoder"


def test_search_explain_records_cross_encoder_rerank_stage(
    monkeypatch, cross_encoder_db
):
    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return [
                0.95 if "target document" in document else 0.10
                for _, document in pairs
            ]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    cross_encoder_db.upsert([
        Document(
            id="d1",
            text="ordinary document",
            vector=[0.1] * 8,
        ),
        Document(
            id="d2",
            text="target document",
            vector=[0.1] * 8,
        ),
    ])

    result = cross_encoder_db.search(
        "find the target",
        top_k=1,
        rerank="cross-encoder",
        explain=True,
    )

    assert isinstance(result, ExplainedSearchResult)
    assert result.results[0].id == "d2"

    explanation = result.explanation

    assert explanation.timings_ms["rerank"] >= 0
    assert explanation.candidate_counts["reranked"] == 1
    assert explanation.candidate_counts["final"] == 1


def test_cross_encoder_rerank_requires_text_query(cross_encoder_db):
    cross_encoder_db.upsert([
        Document(
            id="d1",
            text="some document",
            vector=[0.1] * 8,
        ),
    ])

    with pytest.raises(ConfigurationError, match="requires a text query"):
        cross_encoder_db.search(
            vector=[0.1] * 8,
            top_k=1,
            rerank="cross-encoder",
        )

def test_cross_encoder_rerank_requires_document_text(monkeypatch, cross_encoder_db):
    class FakeCrossEncoder:
        def __init__(self, model_name):
            pass

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    cross_encoder_db.upsert([
        Document(
            id="d1",
            text=None,
            vector=[0.1] * 8,
        ),
    ])

    with pytest.raises(ConfigurationError, match="requires document text"):
        cross_encoder_db.search(
            "find something",
            top_k=1,
            rerank="cross-encoder",
        )


def test_writes_invalidate_query_cache(db):
    db._cache = SemanticCache(threshold=0.99)
    db.upsert([Document(id="1", text="apple pie")], namespace="ns")
    db.upsert([Document(id="o1", text="apple pie")], namespace="other")
    db.search("apple pie", top_k=3, namespace="ns")          # miss -> cached
    db.search("apple pie", top_k=3, namespace="other")       # miss -> cached
    assert db._cache.misses == 2

    # a write to "ns" evicts only that namespace's entries
    db.upsert([Document(id="2", text="apple pie tart")], namespace="ns")
    db.search("apple pie", top_k=3, namespace="other")       # still cached -> hit
    assert db._cache.hits == 1
    res = db.search("apple pie", top_k=3, namespace="ns")    # evicted -> fresh
    assert db._cache.misses == 3
    assert "2" in {r.id for r in res}

    db.update("1", text="apple pie updated", namespace="ns")
    db.search("apple pie", top_k=3, namespace="ns")
    assert db._cache.misses == 4

    db.delete(["1"], namespace="ns")
    db.search("apple pie", top_k=3, namespace="ns")
    assert db._cache.misses == 5


def test_writes_keep_cache_when_invalidation_disabled(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    cfg = DynavecConfig(
        vector_bucket="b", index="i", table="t", dimension=8,
        cache_invalidate_on_write=False,
    )
    db = Dynavec(cfg, embedder=HashEmbedder(8))
    db._cache = SemanticCache(threshold=0.99)

    db.upsert([Document(id="1", text="apple pie")], namespace="ns")
    db.search("apple pie", top_k=3, namespace="ns")          # cached
    db.upsert([Document(id="2", text="apple pie tart")], namespace="ns")
    db.search("apple pie", top_k=3, namespace="ns")          # stale hit
    assert db._cache.hits == 1
