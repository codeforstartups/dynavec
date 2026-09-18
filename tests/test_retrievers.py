"""Offline tests for MultiQueryRetriever and HyDERetriever (#215).

Reuses the in-memory FakeS3 / FakeDDB harness from test_client_inmemory and a
tiny mapping embedder so rankings are fully controlled.
"""

import logging
import threading
import time

import pytest
from test_client_inmemory import FakeDDB, FakeGraph, FakeS3

import dynavec.client as client_mod
from dynavec import Document, Dynavec, DynavecConfig, HyDERetriever, MultiQueryRetriever
from dynavec.embeddings.base import Embedder
from dynavec.exceptions import ConfigurationError
from dynavec.integrations.tools import make_retriever_fn

E0, E1, E2, E3 = ([1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0])

# query text -> vector (anything unknown embeds to a neutral vector)
VECTORS = {
    "orig": E0,
    "alt": E1,
    "alt2": E2,
    "hypo": E1,
}


class MapEmbedder(Embedder):
    dimension = 4

    def __init__(self):
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return [list(VECTORS.get(t, [0.5, 0.5, 0.5, 0.5])) for t in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return list(VECTORS.get(text, [0.5, 0.5, 0.5, 0.5]))


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)
    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=4)
    d = Dynavec(cfg, embedder=MapEmbedder())
    d.upsert(
        [
            Document(id="d0", text="zero", vector=[1.0, 0.0, 0.0, 0.0], metadata={"lang": "en"}),
            Document(id="d1", text="one", vector=[0.9, 0.1, 0.0, 0.0], metadata={"lang": "en"}),
            Document(id="d2", text="two", vector=[0.0, 1.0, 0.0, 0.0], metadata={"lang": "fr"}),
            Document(id="d3", text="three", vector=[0.0, 0.9, 0.1, 0.0], metadata={"lang": "en"}),
            Document(id="d4", text="four", vector=[0.0, 0.0, 1.0, 0.0], metadata={"lang": "en"}),
        ]
    )
    return d


def ids(results):
    return [r.id for r in results]


# ----------------------------------------------------------------- multi-query
def test_multiquery_fuses_lists_and_keeps_original_first_on_ties(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=2, per_query_k=2)
    # orig -> [d0, d1], alt -> [d2, d3]; d0 and d2 tie on RRF, original list wins.
    assert ids(r.search("orig")) == ["d0", "d2"]


def test_per_query_k_is_floored_at_top_k(db):
    seen = []
    real = db.search

    def spy(query=None, **kw):
        seen.append(kw["top_k"])
        return real(query, **kw)

    db.search = spy
    MultiQueryRetriever(db, lambda q: ["alt"], top_k=5, per_query_k=2).search("orig")
    MultiQueryRetriever(db, lambda q: ["alt"], top_k=3).search("orig")
    assert seen == [5, 5, 6, 6]


def test_multiquery_original_weight_reorders(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=2, per_query_k=2, original_weight=3.0)
    assert ids(r.search("orig")) == ["d0", "d1"]  # unweighted this would be d0, d2


def test_multiquery_dedupes_documents_across_lists(db):
    r = MultiQueryRetriever(db, lambda q: ["orig-ish", "alt"], top_k=10, per_query_k=3)
    out = ids(r.search("orig"))
    assert len(out) == len(set(out))


def test_multiquery_include_original_false_skips_original(db):
    calls = []
    real = db.search

    def spy(query=None, **kw):
        calls.append(query)
        return real(query, **kw)

    db.search = spy
    MultiQueryRetriever(db, lambda q: ["alt"], include_original=False).search("orig")
    assert calls == ["alt"]


def test_multiquery_sanitizes_generated_queries(db):
    calls = []
    real = db.search

    def spy(query=None, **kw):
        calls.append(query)
        return real(query, **kw)

    db.search = spy
    gen = lambda q: ["  alt ", "ALT", "", "   ", "ORIG", None, 7, "alt2", "extra"]  # noqa: E731
    MultiQueryRetriever(db, gen, n_queries=2).search("orig")
    # original first, then blanks/dupes/non-strings dropped, capped at n_queries=2
    assert sorted(calls) == ["alt", "alt2", "orig"]  # thread order is not asserted


def test_multiquery_accepts_single_string_output(db):
    r = MultiQueryRetriever(db, lambda q: "alt", top_k=2, per_query_k=2)
    assert ids(r.search("orig")) == ["d0", "d2"]


def test_multiquery_generation_failure_falls_back_and_warns(db, caplog):
    def boom(q):
        raise RuntimeError("llm down")

    r = MultiQueryRetriever(db, boom, top_k=2)
    with caplog.at_level(logging.WARNING, logger="dynavec.retrievers"):
        out = r.search("orig")
    assert ids(out) == ["d0", "d1"]
    assert "falling back" in caplog.text


def test_multiquery_empty_generation_searches_original_even_without_include_original(db):
    r = MultiQueryRetriever(db, lambda q: [], include_original=False, top_k=2)
    assert ids(r.search("orig")) == ["d0", "d1"]


def test_multiquery_on_generate_error_raise(db):
    def boom(q):
        raise RuntimeError("llm down")

    with pytest.raises(RuntimeError):
        MultiQueryRetriever(db, boom, on_generate_error="raise").search("orig")


def test_multiquery_store_errors_propagate(db):
    def broken_search(*a, **kw):
        raise OSError("s3 vectors unavailable")

    db.search = broken_search
    with pytest.raises(OSError):
        MultiQueryRetriever(db, lambda q: ["alt"]).search("orig")


def test_multiquery_scores_are_rrf_scale(db):
    out = MultiQueryRetriever(db, lambda q: ["alt"], rrf_k=60).search("orig")
    assert all(0 < r.score <= 2 / 61 for r in out)


def test_multiquery_result_order_is_independent_of_thread_timing(db):
    real = db.search

    def make_slow(delays):
        def slow(query=None, **kw):
            time.sleep(delays.get(query, 0))
            return real(query, **kw)

        return slow

    r = MultiQueryRetriever(db, lambda q: ["alt", "alt2"], top_k=5, per_query_k=3)
    db.search = make_slow({"orig": 0.15})
    a = ids(r.search("orig"))
    db.search = make_slow({"alt2": 0.15})
    b = ids(r.search("orig"))
    assert a == b


def test_multiquery_runs_subsearches_concurrently(db):
    active, peak, lock = 0, 0, threading.Lock()
    real = db.search

    def tracked(query=None, **kw):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        try:
            return real(query, **kw)
        finally:
            with lock:
                active -= 1

    db.search = tracked
    MultiQueryRetriever(db, lambda q: ["alt", "alt2"]).search("orig")
    assert peak >= 2


def test_filter_is_passed_to_every_subsearch(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=10, per_query_k=10)
    out = r.search("orig", filter={"lang": "en"})
    assert "d2" not in ids(out)  # d2 is lang=fr
    assert set(ids(out)) <= {"d0", "d1", "d3", "d4"}


def test_namespace_view_and_namespace_kwarg(db):
    db.upsert([Document(id="k0", text="kb doc", vector=E0)], namespace="kb")
    via_view = MultiQueryRetriever(db.namespace("kb"), lambda q: ["alt"])
    via_kwarg = MultiQueryRetriever(db, lambda q: ["alt"], namespace="kb")
    assert ids(via_view.search("orig")) == ["k0"]
    assert ids(via_kwarg.search("orig")) == ["k0"]
    # default namespace is untouched by the kb doc
    assert "k0" not in ids(MultiQueryRetriever(db, lambda q: ["alt"]).search("orig"))


def test_per_call_top_k_override(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=4)
    assert len(r.search("orig", top_k=1)) == 1


async def test_asearch_matches_search(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=2, per_query_k=2)
    assert ids(await r.asearch("orig")) == ids(r.search("orig"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"per_query_k": 0},
        {"rrf_k": 0},
        {"original_weight": 0},
        {"n_queries": 0},
        {"on_generate_error": "ignore"},
    ],
)
def test_multiquery_validates_arguments(db, kwargs):
    with pytest.raises(ValueError):
        MultiQueryRetriever(db, lambda q: [], **kwargs)


def test_rejects_blank_query_and_non_callable(db):
    r = MultiQueryRetriever(db, lambda q: [])
    with pytest.raises(ValueError):
        r.search("   ")
    with pytest.raises(TypeError):
        MultiQueryRetriever(db, "not callable")


# ------------------------------------------------------------------------ HyDE
def test_hyde_embeds_hypothetical_as_document_and_fuses_with_query(db):
    emb = db.embedder
    r = HyDERetriever(db, lambda q: "hypo", top_k=2, per_query_k=2)
    out = r.search("orig")
    # "hypo" is embedded on the document side, the raw query on the query side
    assert ["hypo"] in emb.doc_calls
    assert "orig" in emb.query_calls and "hypo" not in emb.query_calls
    # orig -> [d0, d1]; hypo (E1) -> [d2, d3]; tie between d0 and d2 goes to original
    assert ids(out) == ["d0", "d2"]


def test_hyde_without_original_uses_only_hypothetical(db):
    r = HyDERetriever(db, lambda q: "hypo", include_original=False, top_k=2)
    assert ids(r.search("orig")) == ["d2", "d3"]
    assert "orig" not in db.embedder.query_calls


@pytest.mark.parametrize("bad", ["", "   ", None, []])
def test_hyde_blank_generation_falls_back_to_original(db, bad):
    r = HyDERetriever(db, lambda q: bad, include_original=False, top_k=2)
    assert ids(r.search("orig")) == ["d0", "d1"]


def test_hyde_generation_exception_falls_back(db):
    def boom(q):
        raise TimeoutError("slow llm")

    assert ids(HyDERetriever(db, boom, top_k=2).search("orig")) == ["d0", "d1"]
    with pytest.raises(TimeoutError):
        HyDERetriever(db, boom, on_generate_error="raise").search("orig")


def test_hyde_requires_embedder(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)
    bare = Dynavec(DynavecConfig(vector_bucket="b", index="i", table="t", dimension=4))
    with pytest.raises(ConfigurationError):
        HyDERetriever(bare, lambda q: "x")


def test_hyde_honors_filter_and_namespace_view(db):
    db.upsert([Document(id="k0", text="kb", vector=E1, metadata={"lang": "en"})], namespace="kb")
    r = HyDERetriever(db.namespace("kb"), lambda q: "hypo", top_k=5)
    assert ids(r.search("orig", filter={"lang": "en"})) == ["k0"]


async def test_hyde_asearch(db):
    r = HyDERetriever(db, lambda q: "hypo", top_k=2, per_query_k=2)
    assert ids(await r.asearch("orig")) == ids(r.search("orig"))


# ------------------------------------------------------------- tool factory
def test_make_retriever_fn_accepts_retrievers(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=4, per_query_k=2)
    fn = make_retriever_fn(r, top_k=2, join=" | ")
    assert fn("orig") == "zero | two"


def test_make_retriever_fn_rejects_rescore_for_retrievers(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"])
    with pytest.raises(ValueError):
        make_retriever_fn(r, rescore="cosine")
