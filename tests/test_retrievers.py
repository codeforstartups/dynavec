import asyncio
import logging
import threading
import time

import pytest
from test_client_inmemory import FakeDDB, FakeGraph, FakeS3

import dynavec.client as client_mod
from dynavec import (
    Document,
    Dynavec,
    DynavecConfig,
    HyDERetriever,
    MultiQueryRetriever,
    QueryExpansionRetriever,
)
from dynavec.embeddings.base import Embedder
from dynavec.exceptions import ConfigurationError
from dynavec.integrations.tools import make_retriever_fn
from dynavec.models import SearchResult

# Test embeddings in 4D
E0 = [1.0, 0.0, 0.0, 0.0]
E1 = [0.0, 1.0, 0.0, 0.0]
E2 = [0.0, 0.0, 1.0, 0.0]
E3 = [0.0, 0.0, 0.0, 1.0]


class FakeEmbedder(Embedder):
    dimension = 4

    def __init__(self):
        self.query_calls: list[str] = []
        self.doc_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        if "alt" in text:
            return E1
        if "hypo" in text:
            return E1
        return E0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        res = []
        for t in texts:
            if "hypo1" in t:
                res.append(E1)
            elif "hypo2" in t:
                res.append(E2)
            elif "alt" in t or "hypo" in t:
                res.append(E1)
            else:
                res.append(E0)
        return res


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=4, region="us-east-1")
    client = Dynavec(cfg, embedder=FakeEmbedder())

    # Seed 4 documents:
    # d0: near E0, text="zero"
    # d1: near E0, text="one"
    # d2: near E1, text="two", metadata={"lang": "fr"}
    # d3: near E1, text="three", metadata={"lang": "en"}
    docs = [
        Document(id="d0", text="zero", vector=E0, metadata={"lang": "en"}),
        Document(id="d1", text="one", vector=E0, metadata={"lang": "en"}),
        Document(id="d2", text="two", vector=E1, metadata={"lang": "fr"}),
        Document(id="d3", text="three", vector=E1, metadata={"lang": "en"}),
    ]
    client.upsert(docs)
    return client


def ids(results: list[SearchResult]) -> list[str]:
    return [r.id for r in results]


# ============================================================================
# MultiQueryRetriever Tests
# ============================================================================


def test_multiquery_merges_unique_ids(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=4, per_query_k=2)
    out = r.search("orig")
    # orig matches [d0, d1], alt matches [d2, d3] -> fused contains all 4
    assert len(out) == 4
    assert set(ids(out)) == {"d0", "d1", "d2", "d3"}
    # Results must have unique IDs
    assert len(ids(out)) == len(set(ids(out)))


def test_multiquery_llm_generate_queries_kwarg(db):
    """Verify Issue #215 spec parameter name 'llm_generate_queries' works."""
    r = MultiQueryRetriever(db, llm_generate_queries=lambda q: ["alt"], top_k=2)
    out = r.search("orig")
    assert len(out) == 2


def test_multiquery_include_original_false_skips_original(db):
    calls = []
    real = db.search

    def spy(query=None, **kw):
        calls.append(query)
        return real(query, **kw)

    db.search = spy
    MultiQueryRetriever(db, lambda q: ["alt"], include_original=False).search("orig")
    assert calls == ["alt"]


def test_multiquery_include_original_query_alias(db):
    """Verify Issue #215 spec parameter name 'include_original_query' works."""
    calls = []
    real = db.search

    def spy(query=None, **kw):
        calls.append(query)
        return real(query, **kw)

    db.search = spy
    MultiQueryRetriever(db, lambda q: ["alt"], include_original_query=False).search("orig")
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
    assert sorted(calls) == ["alt", "alt2", "orig"]


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


def test_multiquery_custom_weights(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=2)
    # With weights [0.1, 10.0], alt results (d2, d3) should dominate over orig (d0, d1)
    res = r.search("orig", weights=[0.1, 10.0])
    assert ids(res)[0] in {"d2", "d3"}


def test_filter_is_passed_to_every_subsearch(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=10, per_query_k=10)
    out = r.search("orig", filter={"lang": "en"})
    assert "d2" not in ids(out)  # d2 is lang=fr
    assert set(ids(out)) <= {"d0", "d1", "d3"}


def test_namespace_view_and_namespace_kwarg(db):
    db.upsert([Document(id="k0", text="kb doc", vector=E0)], namespace="kb")
    via_view = MultiQueryRetriever(db.namespace("kb"), lambda q: ["alt"])
    via_kwarg = MultiQueryRetriever(db, lambda q: ["alt"], namespace="kb")
    assert ids(via_view.search("orig")) == ["k0"]
    assert ids(via_kwarg.search("orig")) == ["k0"]
    assert "k0" not in ids(MultiQueryRetriever(db, lambda q: ["alt"]).search("orig"))


def test_per_call_top_k_override(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=4)
    assert len(r.search("orig", top_k=1)) == 1


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


# ============================================================================
# HyDERetriever Tests
# ============================================================================


def test_hyde_embeds_hypothetical_as_document_and_fuses_with_query(db):
    emb = db.embedder
    r = HyDERetriever(db, lambda q: "hypo", top_k=2, per_query_k=2)
    out = r.search("orig")
    # "hypo" is embedded on the document side, raw query on the query side
    assert ["hypo"] in emb.doc_calls
    assert "orig" in emb.query_calls and "hypo" not in emb.query_calls
    # orig -> [d0, d1]; hypo (E1) -> [d2, d3]; tie goes to original
    assert ids(out) == ["d0", "d2"]


def test_hyde_llm_generate_hypothetical_kwarg(db):
    """Verify Issue #215 spec parameter name 'llm_generate_hypothetical' works."""
    r = HyDERetriever(db, llm_generate_hypothetical=lambda q: "hypo", top_k=2)
    assert len(r.search("orig")) == 2


def test_hyde_without_original_uses_only_hypothetical(db):
    r = HyDERetriever(db, lambda q: "hypo", include_original=False, top_k=2)
    assert ids(r.search("orig")) == ["d2", "d3"]
    assert "orig" not in db.embedder.query_calls


def test_hyde_include_original_query_alias(db):
    """Verify Issue #215 spec parameter name 'include_original_query' works."""
    r = HyDERetriever(db, lambda q: "hypo", include_original_query=False, top_k=2)
    assert ids(r.search("orig")) == ["d2", "d3"]


def test_hyde_multi_passage_centroid_average_strategy(db):
    """Multiple hypothetical passages should be averaged into a single centroid vector."""
    # hypo1 -> E1, hypo2 -> E2
    # centroid of E1 and E2 normalized: [0, 1/sqrt(2), 1/sqrt(2), 0]
    r = HyDERetriever(
        db,
        lambda q: ["hypo1", "hypo2"],
        strategy="average",
        include_original=False,
        top_k=2,
    )
    out = r.search("orig")
    assert ["hypo1", "hypo2"] in db.embedder.doc_calls
    assert len(out) == 2


def test_hyde_multi_passage_fuse_strategy(db):
    """strategy='fuse' executes separate vector searches and fuses via RRF."""
    r = HyDERetriever(
        db,
        lambda q: ["hypo1", "hypo2"],
        strategy="fuse",
        include_original=False,
        top_k=4,
    )
    out = r.search("orig")
    assert len(out) >= 2


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
    bare = Dynavec(DynavecConfig(vector_bucket="b", index="i", table="t", dimension=4, region="us-east-1"))
    with pytest.raises(ConfigurationError):
        HyDERetriever(bare, lambda q: "x")


def test_hyde_validates_strategy(db):
    with pytest.raises(ValueError, match="strategy"):
        HyDERetriever(db, lambda q: "x", strategy="invalid")


def test_hyde_honors_filter_and_namespace_view(db):
    db.upsert([Document(id="k0", text="kb", vector=E1, metadata={"lang": "en"})], namespace="kb")
    r = HyDERetriever(db.namespace("kb"), lambda q: "hypo", top_k=5)
    assert ids(r.search("orig", filter={"lang": "en"})) == ["k0"]


# ============================================================================
# Async Support Tests
# ============================================================================


async def test_multiquery_asearch_matches_search(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=2, per_query_k=2)
    assert ids(await r.asearch("orig")) == ids(r.search("orig"))


async def test_hyde_asearch(db):
    r = HyDERetriever(db, lambda q: "hypo", top_k=2, per_query_k=2)
    assert ids(await r.asearch("orig")) == ids(r.search("orig"))


async def test_multiquery_async_generator_in_asearch(db):
    """Verify that an async def generator is properly awaited in asearch."""

    async def async_gen(q: str):
        await asyncio.sleep(0.01)
        return ["alt"]

    r = MultiQueryRetriever(db, async_gen, top_k=4)
    out = await r.asearch("orig")
    assert "d2" in ids(out)


def test_multiquery_async_generator_in_sync_search(db):
    """Verify that an async def generator also works when called from sync search()."""

    async def async_gen(q: str):
        await asyncio.sleep(0.01)
        return ["alt"]

    r = MultiQueryRetriever(db, async_gen, top_k=4)
    out = r.search("orig")
    assert "d2" in ids(out)


# ============================================================================
# Dynavec and NamespaceView Factory Methods
# ============================================================================


def test_dynavec_factory_methods(db):
    multi = db.as_multiquery_retriever(lambda q: ["alt"], top_k=2)
    assert isinstance(multi, MultiQueryRetriever)
    assert len(multi.search("orig")) == 2

    hyde = db.as_hyde_retriever(lambda q: "hypo", top_k=2)
    assert isinstance(hyde, HyDERetriever)
    assert len(hyde.search("orig")) == 2


def test_namespace_view_factory_methods(db):
    db.upsert([Document(id="k0", text="kb doc", vector=E0)], namespace="kb")
    kb = db.namespace("kb")

    multi = kb.as_multiquery_retriever(lambda q: ["alt"])
    assert isinstance(multi, MultiQueryRetriever)
    assert ids(multi.search("orig")) == ["k0"]

    hyde = kb.as_hyde_retriever(lambda q: "hypo")
    assert isinstance(hyde, HyDERetriever)
    assert ids(hyde.search("orig")) == ["k0"]


# ============================================================================
# Re-exports and Module Imports
# ============================================================================


def test_imports_from_dynavec():
    import dynavec

    assert hasattr(dynavec, "MultiQueryRetriever")
    assert hasattr(dynavec, "HyDERetriever")
    assert hasattr(dynavec, "QueryExpansionRetriever")


def test_imports_from_dynavec_retrieval():
    from dynavec.retrieval import HyDERetriever as H
    from dynavec.retrieval import MultiQueryRetriever as M
    from dynavec.retrieval import QueryExpansionRetriever as Q

    assert M is MultiQueryRetriever
    assert H is HyDERetriever
    assert Q is QueryExpansionRetriever


# ============================================================================
# Tool Factory Tests
# ============================================================================


def test_make_retriever_fn_accepts_retrievers(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"], top_k=4, per_query_k=2)
    fn = make_retriever_fn(r, top_k=2, join=" | ")
    assert fn("orig") == "zero | two"


def test_make_retriever_fn_rejects_rescore_for_retrievers(db):
    r = MultiQueryRetriever(db, lambda q: ["alt"])
    with pytest.raises(ValueError, match="rescore is not supported"):
        make_retriever_fn(r, rescore="cosine")
