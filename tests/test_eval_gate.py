"""Tests for eval regression gating and CI checks."""

from __future__ import annotations

import json

import pytest

from dynavec.eval import (
    MetricDirection,
    assert_no_regression,
    check_regression,
)
from dynavec.eval.gate import get_metric_direction
from dynavec.eval.retrieval import RetrievalEvalSummary
from dynavec.eval.runner import EvalSummary


def _retrieval_summary(recall_10: float, ndcg_10: float, mrr: float = 0.8, p95_latency: float = 30.0) -> RetrievalEvalSummary:
    return RetrievalEvalSummary(
        total_queries=10,
        k_values=[5, 10],
        mean_recall={5: recall_10 * 0.8, 10: recall_10},
        mean_mrr=mrr,
        mean_ndcg={5: ndcg_10 * 0.9, 10: ndcg_10},
        mean_precision={5: 0.5, 10: 0.4},
        p50_latency_ms=12.0,
        p95_latency_ms=p95_latency,
    )


def _rag_summary(faithfulness: float, relevance: float = 0.85, p95_latency: float = 50.0) -> EvalSummary:
    return EvalSummary(
        total_samples=20,
        mean_faithfulness=faithfulness,
        mean_relevance=relevance,
        pass_rate=0.9,
        pass_threshold=0.7,
        p95_latency_ms=p95_latency,
    )


# ---------------------------------------------------------------------------
# 1. Basic Pass / Fail
# ---------------------------------------------------------------------------

def test_identical_runs_pass():
    base = _retrieval_summary(0.85, 0.80)
    curr = _retrieval_summary(0.85, 0.80)
    report = check_regression(curr, base)
    assert report.passed
    assert len(report.regressions) == 0


def test_improved_runs_pass():
    base = _retrieval_summary(0.80, 0.75, p95_latency=40.0)
    curr = _retrieval_summary(0.90, 0.85, p95_latency=30.0)
    report = check_regression(curr, base)
    assert report.passed
    assert report.diffs["recall@10"].status == "IMPROVED"
    assert report.diffs["p95_latency_ms"].status == "IMPROVED"


def test_higher_is_better_drop_triggers_regression():
    base = {"recall@10": 0.90}
    curr = {"recall@10": 0.80}
    report = check_regression(curr, base, max_drop=0.05)
    assert not report.passed
    assert len(report.regressions) == 1
    assert report.regressions[0].metric == "recall@10"
    assert report.regressions[0].is_regression is True
    assert report.regressions[0].status == "REGRESSION"


def test_higher_is_better_within_threshold_passes():
    base = {"recall@10": 0.90}
    curr = {"recall@10": 0.88}
    report = check_regression(curr, base, max_drop=0.05)
    assert report.passed
    assert len(report.regressions) == 0


# ---------------------------------------------------------------------------
# 2. Lower-Is-Better Polarity (Latency, Error Rate)
# ---------------------------------------------------------------------------

def test_latency_increase_triggers_regression():
    base = {"p95_latency_ms": 100.0}
    curr = {"p95_latency_ms": 120.0}
    report = check_regression(curr, base, max_drop=0.10)
    assert not report.passed
    assert len(report.regressions) == 1
    assert report.regressions[0].metric == "p95_latency_ms"
    assert report.regressions[0].direction == MetricDirection.LOWER_IS_BETTER
    assert report.regressions[0].status == "REGRESSION"


def test_latency_reduction_is_improvement():
    base = {"p95_latency_ms": 100.0}
    curr = {"p95_latency_ms": 75.0}
    report = check_regression(curr, base, max_drop=0.05)
    assert report.passed
    assert report.diffs["p95_latency_ms"].status == "IMPROVED"


def test_compound_latency_detected_as_lower_is_better():
    assert get_metric_direction("p99_latency_ms") == MetricDirection.LOWER_IS_BETTER
    assert get_metric_direction("mean_latency_ms") == MetricDirection.LOWER_IS_BETTER
    assert get_metric_direction("retrieval_duration_seconds") == MetricDirection.LOWER_IS_BETTER
    assert get_metric_direction("system_error_rate") == MetricDirection.LOWER_IS_BETTER
    assert get_metric_direction("total_cost") == MetricDirection.LOWER_IS_BETTER
    assert get_metric_direction("accuracy") == MetricDirection.HIGHER_IS_BETTER


# ---------------------------------------------------------------------------
# 3. Per-Metric Threshold Overrides
# ---------------------------------------------------------------------------

def test_per_metric_threshold_overrides():
    base = {"recall@10": 0.90, "ndcg@10": 0.80}
    curr = {"recall@10": 0.86, "ndcg@10": 0.75}

    report = check_regression(
        curr,
        base,
        max_drop=0.10,
        metric_thresholds={"ndcg@10": 0.03},
    )
    assert not report.passed
    assert len(report.regressions) == 1
    assert report.regressions[0].metric == "ndcg@10"
    assert report.diffs["recall@10"].status == "PASS"


# ---------------------------------------------------------------------------
# 4. Absolute vs Relative Drop
# ---------------------------------------------------------------------------

def test_absolute_drop_mode():
    base = {"faithfulness": 0.50}
    curr = {"faithfulness": 0.44}

    report_rel = check_regression(curr, base, max_drop=0.10, relative=True)
    assert not report_rel.passed

    report_abs = check_regression(curr, base, max_drop=0.07, relative=False)
    assert report_abs.passed


# ---------------------------------------------------------------------------
# 5. Edge Cases (Zero Baseline, NaN, Zero Common Metrics, Epsilon)
# ---------------------------------------------------------------------------

def test_zero_baseline_does_not_crash_division():
    base = {"mrr": 0.0}
    curr = {"mrr": 0.0}
    report = check_regression(curr, base, max_drop=0.05)
    assert report.passed
    assert report.diffs["mrr"].relative_change == 0.0


def test_zero_baseline_drop_detection():
    base = {"mrr": 0.0}
    curr = {"mrr": 0.1}
    report = check_regression(curr, base, max_drop=0.05)
    assert report.passed
    assert report.diffs["mrr"].status == "IMPROVED"


def test_floating_point_epsilon_precision():
    base = {"score": 0.85}
    curr = {"score": 0.8075}
    report = check_regression(curr, base, max_drop=0.05)
    assert report.passed


def test_nan_metric_triggers_regression():
    base = {"recall@10": 0.85}
    curr = {"recall@10": float("nan")}
    report = check_regression(curr, base, max_drop=0.05)
    assert not report.passed
    assert len(report.regressions) == 1
    assert "NaN" in report.regressions[0].status

    # Baseline is NaN
    base_nan = {"recall@10": float("nan")}
    curr_valid = {"recall@10": 0.85}
    report_base_nan = check_regression(curr_valid, base_nan, max_drop=0.05)
    assert not report_base_nan.passed
    assert "NaN" in report_base_nan.regressions[0].status


def test_zero_common_metrics_fails_regression_by_default():
    base = {"retrieval_recall": 0.80}
    curr = {"rag_faithfulness": 0.90}
    report = check_regression(curr, base)
    assert not report.passed
    assert "FAILED" in report.summary_table()


def test_zero_common_metrics_passes_when_allow_empty():
    base = {"retrieval_recall": 0.80}
    curr = {"rag_faithfulness": 0.90}
    report = check_regression(curr, base, allow_empty=True)
    assert report.passed


def test_fail_on_missing_metric_flag():
    base = {"recall@10": 0.90}
    curr = {"faithfulness": 0.85}

    report = check_regression(curr, base, max_drop=0.05, allow_empty=True)
    assert report.passed

    report_strict = check_regression(
        curr,
        base,
        metric_thresholds={"recall@10": 0.05},
        fail_on_missing_metric=True,
    )
    assert not report_strict.passed
    assert len(report_strict.regressions) == 1
    assert "missing" in report_strict.regressions[0].status


# ---------------------------------------------------------------------------
# 6. JSON Files & Dict Snapshots
# ---------------------------------------------------------------------------

def test_check_regression_from_json_files(tmp_path):
    base_file = tmp_path / "base.json"
    curr_file = tmp_path / "curr.json"

    base_summary = _rag_summary(0.95, 0.90)
    curr_summary = _rag_summary(0.70, 0.90)

    base_file.write_text(json.dumps(base_summary.to_dict()), encoding="utf-8")
    curr_file.write_text(json.dumps(curr_summary.to_dict()), encoding="utf-8")

    report = check_regression(curr_file, base_file, max_drop=0.05)
    assert not report.passed
    assert len(report.regressions) == 1
    assert report.regressions[0].metric == "faithfulness"


# ---------------------------------------------------------------------------
# 7. Summary Table & Assert Helper
# ---------------------------------------------------------------------------

def test_summary_table_formatting():
    base = {"recall@10": 0.90, "p95_latency_ms": 50.0}
    curr = {"recall@10": 0.80, "p95_latency_ms": 60.0}
    report = check_regression(curr, base, max_drop=0.05)

    table = report.summary_table()
    assert "Metric" in table
    assert "recall@10" in table
    assert "p95_latency_ms" in table
    assert "REGRESSION" in table
    assert "FAILED:" in table


def test_summary_table_formats_nan_cleanly():
    base = {"recall@10": 0.90}
    curr = {"recall@10": float("nan")}
    report = check_regression(curr, base, max_drop=0.05)
    table = report.summary_table()
    assert "NaN" in table


def test_assert_no_regression_raises():
    base = {"recall@10": 0.90}
    curr = {"recall@10": 0.70}

    with pytest.raises(AssertionError, match="Eval metric regression detected"):
        assert_no_regression(curr, base, max_drop=0.05)


def test_assert_no_regression_passes():
    base = {"recall@10": 0.90}
    curr = {"recall@10": 0.92}
    assert_no_regression(curr, base, max_drop=0.05)
