"""Tests for EvalRunStore — trend tracking, tag filtering, and run comparison."""

from __future__ import annotations

import pytest

from dynavec.eval import EvalRun, EvalRunStore
from dynavec.eval.retrieval import RetrievalEvalSummary
from dynavec.eval.runner import EvalSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieval_summary(recall_10: float, ndcg_10: float, mrr: float = 0.8) -> RetrievalEvalSummary:
    return RetrievalEvalSummary(
        total_queries=10,
        k_values=[5, 10],
        mean_recall={5: recall_10 * 0.8, 10: recall_10},
        mean_mrr=mrr,
        mean_ndcg={5: ndcg_10 * 0.9, 10: ndcg_10},
        mean_precision={5: 0.5, 10: 0.4},
        p50_latency_ms=12.0,
        p95_latency_ms=30.0,
    )


def _rag_summary(faithfulness: float, relevance: float = 0.85) -> EvalSummary:
    return EvalSummary(
        total_samples=20,
        mean_faithfulness=faithfulness,
        mean_relevance=relevance,
        pass_rate=0.9,
        pass_threshold=0.7,
        p95_latency_ms=50.0,
    )


# ---------------------------------------------------------------------------
# 1. record()
# ---------------------------------------------------------------------------

def test_record_retrieval_summary_returns_eval_run():
    store = EvalRunStore()
    summary = _retrieval_summary(0.85, 0.80)
    run = store.record(summary, tags={"dataset": "squad", "model": "ada-002"})

    assert isinstance(run, EvalRun)
    assert run.run_id
    assert run.tags == {"dataset": "squad", "model": "ada-002"}
    assert run.metrics["recall@10"] == pytest.approx(0.85)
    assert run.metrics["ndcg@10"] == pytest.approx(0.80)
    assert run.metrics["mrr"] == pytest.approx(0.8)


def test_record_rag_summary_stores_faithfulness():
    store = EvalRunStore()
    run = store.record(_rag_summary(0.92), tags={"model": "gpt-4o"})

    assert run.metrics["faithfulness"] == pytest.approx(0.92)
    assert run.metrics["relevance"] == pytest.approx(0.85)


def test_record_unsupported_type_raises():
    store = EvalRunStore()
    with pytest.raises(TypeError):
        store.record("not a summary")  # type: ignore[arg-type]


def test_record_respects_max_runs():
    store = EvalRunStore(max_runs=3)
    for i in range(5):
        store.record(_retrieval_summary(0.5 + i * 0.05, 0.6))
    assert len(store.all_runs()) == 3


# ---------------------------------------------------------------------------
# 2. filter()
# ---------------------------------------------------------------------------

def test_filter_by_single_tag():
    store = EvalRunStore()
    store.record(_retrieval_summary(0.80, 0.75), tags={"dataset": "squad", "model": "ada"})
    store.record(_retrieval_summary(0.70, 0.65), tags={"dataset": "nq",    "model": "ada"})
    store.record(_retrieval_summary(0.90, 0.85), tags={"dataset": "squad", "model": "v3"})

    squad_runs = store.filter(dataset="squad")
    assert len(squad_runs) == 2
    assert all(r.tags["dataset"] == "squad" for r in squad_runs)


def test_filter_by_multiple_tags():
    store = EvalRunStore()
    store.record(_retrieval_summary(0.80, 0.75), tags={"dataset": "squad", "model": "ada"})
    store.record(_retrieval_summary(0.70, 0.65), tags={"dataset": "squad", "model": "v3"})

    runs = store.filter(dataset="squad", model="ada")
    assert len(runs) == 1
    assert runs[0].metrics["recall@10"] == pytest.approx(0.80)


def test_filter_no_match_returns_empty():
    store = EvalRunStore()
    store.record(_retrieval_summary(0.80, 0.75), tags={"dataset": "squad"})
    assert store.filter(dataset="beir") == []


# ---------------------------------------------------------------------------
# 3. trend()
# ---------------------------------------------------------------------------

def test_trend_recall_returns_sorted_time_series():
    store = EvalRunStore()
    # record three runs at explicit timestamps
    store.record(_retrieval_summary(0.70, 0.65), ts=1000.0)
    store.record(_retrieval_summary(0.80, 0.75), ts=2000.0)
    store.record(_retrieval_summary(0.90, 0.85), ts=3000.0)

    series = store.trend("recall", k=10)

    assert len(series) == 3
    # oldest first
    assert series[0] == pytest.approx((1000.0, 0.70))
    assert series[1] == pytest.approx((2000.0, 0.80))
    assert series[2] == pytest.approx((3000.0, 0.90))


def test_trend_faithfulness_from_rag_summaries():
    store = EvalRunStore()
    store.record(_rag_summary(0.75), ts=100.0)
    store.record(_rag_summary(0.85), ts=200.0)

    series = store.trend("faithfulness")
    assert len(series) == 2
    assert series[0][1] == pytest.approx(0.75)
    assert series[1][1] == pytest.approx(0.85)


def test_trend_with_tag_filter():
    store = EvalRunStore()
    store.record(_retrieval_summary(0.60, 0.55), tags={"model": "ada"}, ts=100.0)
    store.record(_retrieval_summary(0.80, 0.75), tags={"model": "v3"},  ts=200.0)
    store.record(_retrieval_summary(0.65, 0.60), tags={"model": "ada"}, ts=300.0)

    series = store.trend("recall", k=10, model="ada")
    assert len(series) == 2
    assert series[0][1] == pytest.approx(0.60)
    assert series[1][1] == pytest.approx(0.65)


def test_trend_skips_runs_without_metric():
    store = EvalRunStore()
    # RAG summary has no recall@10
    store.record(_rag_summary(0.9), ts=100.0)
    store.record(_retrieval_summary(0.85, 0.80), ts=200.0)

    series = store.trend("recall", k=10)
    assert len(series) == 1  # only the retrieval run has recall@10
    assert series[0][1] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 4. compare()
# ---------------------------------------------------------------------------

def test_compare_two_runs_returns_both():
    store = EvalRunStore()
    r1 = store.record(_retrieval_summary(0.75, 0.70), tags={"model": "ada"})
    r2 = store.record(_retrieval_summary(0.90, 0.88), tags={"model": "v3"})

    diff = store.compare([r1.run_id, r2.run_id])

    assert set(diff.keys()) == {r1.run_id, r2.run_id}
    assert diff[r1.run_id]["metrics"]["recall@10"] == pytest.approx(0.75)
    assert diff[r2.run_id]["metrics"]["recall@10"] == pytest.approx(0.90)


def test_compare_missing_id_raises():
    store = EvalRunStore()
    run = store.record(_retrieval_summary(0.80, 0.75))
    with pytest.raises(KeyError, match="not found"):
        store.compare([run.run_id, "doesnotexist"])


# ---------------------------------------------------------------------------
# 5. to_dict round-trip
# ---------------------------------------------------------------------------

def test_eval_run_to_dict_is_serialisable():
    import json
    store = EvalRunStore()
    run = store.record(_retrieval_summary(0.82, 0.78), tags={"dataset": "squad"})
    d = run.to_dict()
    json.dumps(d)  # must not raise
    assert d["tags"]["dataset"] == "squad"
    assert "recall@10" in d["metrics"]
