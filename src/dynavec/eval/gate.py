"""CI regression gating and run comparison for evaluation metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .retrieval import RetrievalEvalSummary
from .runner import EvalSummary
from .trend import EvalRun, _metrics_from_rag, _metrics_from_retrieval


class MetricDirection(str, Enum):
    """Directionality of metric quality."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# Known lower-is-better metrics (where an increase indicates degradation/regression)
LOWER_IS_BETTER_METRICS = {
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_latency_ms",
    "latency",
    "latency_ms",
    "error_rate",
    "token_count",
    "cost",
}


def get_metric_direction(
    metric_name: str,
    overrides: dict[str, MetricDirection] | None = None,
) -> MetricDirection:
    """Return whether higher or lower values indicate better quality for a metric."""
    if overrides and metric_name in overrides:
        return overrides[metric_name]
    name_lower = metric_name.lower()
    if (
        name_lower in LOWER_IS_BETTER_METRICS
        or "latency" in name_lower
        or "duration" in name_lower
        or "error_rate" in name_lower
        or "cost" in name_lower
    ):
        return MetricDirection.LOWER_IS_BETTER
    return MetricDirection.HIGHER_IS_BETTER


@dataclass
class MetricDiff:
    """Comparison of a single metric between baseline and current runs."""

    metric: str
    baseline: float
    current: float
    delta: float
    relative_change: float
    threshold: float
    direction: MetricDirection
    is_regression: bool
    status: str  # "PASS", "REGRESSION", "IMPROVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "delta": round(self.delta, 6) if not math.isnan(self.delta) else None,
            "relative_change": round(self.relative_change, 6) if not math.isnan(self.relative_change) else None,
            "threshold": self.threshold,
            "direction": self.direction.value,
            "is_regression": self.is_regression,
            "status": self.status,
        }


@dataclass
class RegressionReport:
    """Aggregated regression analysis between two eval runs."""

    passed: bool
    diffs: dict[str, MetricDiff]
    regressions: list[MetricDiff]
    baseline_id: str | None = None
    current_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "regressions_count": len(self.regressions),
            "regressions": [r.to_dict() for r in self.regressions],
            "diffs": {k: v.to_dict() for k, v in self.diffs.items()},
        }

    def summary_table(self) -> str:
        """Render a formatted ASCII/Markdown table of the regression comparison."""
        if not self.diffs:
            verdict = "PASSED: Zero metrics compared (empty runs)." if self.passed else (
                "FAILED: No common metrics found between baseline and current runs."
            )
            return f"No common metrics evaluated.\n\nOverall: {verdict}"

        headers = ["Metric", "Baseline", "Current", "Delta", "% Change", "Threshold", "Status"]
        rows: list[list[str]] = []

        for name, diff in sorted(self.diffs.items()):
            b_str = f"{diff.baseline:.4f}" if not math.isnan(diff.baseline) else "NaN"
            c_str = f"{diff.current:.4f}" if not math.isnan(diff.current) else "NaN"
            d_str = f"{diff.delta:+.4f}" if not math.isnan(diff.delta) else "NaN"
            pct = f"{diff.relative_change * 100:+.2f}%" if not math.isnan(diff.relative_change) else "N/A"
            rows.append([
                name,
                b_str,
                c_str,
                d_str,
                pct,
                f"<= {diff.threshold * 100:.1f}%" if diff.threshold else "0.0%",
                diff.status,
            ])

        # Compute column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def fmt_row(items: list[str]) -> str:
            return " | ".join(item.ljust(col_widths[i]) for i, item in enumerate(items))

        sep = "-+-".join("-" * w for w in col_widths)
        table_lines = [
            fmt_row(headers),
            sep,
            *(fmt_row(r) for r in rows),
        ]

        verdict = "PASSED: No metric regressions detected." if self.passed else (
            f"FAILED: {len(self.regressions)} metric(s) regressed beyond threshold."
        )
        return "\n".join(table_lines) + f"\n\nOverall: {verdict}"


def _extract_metrics(run_or_summary: Any) -> tuple[dict[str, float], str | None]:
    """Extract a flat metric dict and optional run_id from various summary representations."""
    if isinstance(run_or_summary, (str, Path)):
        path = Path(run_or_summary).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Eval summary file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _extract_metrics(data)

    if isinstance(run_or_summary, RetrievalEvalSummary):
        return _metrics_from_retrieval(run_or_summary), None

    if isinstance(run_or_summary, EvalSummary):
        return _metrics_from_rag(run_or_summary), None

    if isinstance(run_or_summary, EvalRun):
        return dict(run_or_summary.metrics), run_or_summary.run_id

    if isinstance(run_or_summary, dict):
        run_id = run_or_summary.get("run_id")
        # Check if it's an EvalRun dict
        if "metrics" in run_or_summary and isinstance(run_or_summary["metrics"], dict):
            return dict(run_or_summary["metrics"]), str(run_id) if run_id else None

        # Check if it's a serialized RetrievalEvalSummary
        if "mean_recall" in run_or_summary:
            out: dict[str, float] = {}
            if "total_queries" in run_or_summary:
                out["total_queries"] = float(run_or_summary["total_queries"])
            if "mean_mrr" in run_or_summary:
                out["mrr"] = float(run_or_summary["mean_mrr"])
            if "p50_latency_ms" in run_or_summary:
                out["p50_latency_ms"] = float(run_or_summary["p50_latency_ms"])
            if "p95_latency_ms" in run_or_summary:
                out["p95_latency_ms"] = float(run_or_summary["p95_latency_ms"])
            for k, v in run_or_summary.get("mean_recall", {}).items():
                out[f"recall@{k}"] = float(v)
            for k, v in run_or_summary.get("mean_ndcg", {}).items():
                out[f"ndcg@{k}"] = float(v)
            for k, v in run_or_summary.get("mean_precision", {}).items():
                out[f"precision@{k}"] = float(v)
            return out, str(run_id) if run_id else None

        # Check if it's a serialized EvalSummary
        if "mean_faithfulness" in run_or_summary:
            out = {
                "faithfulness": float(run_or_summary.get("mean_faithfulness", 0.0)),
                "relevance": float(run_or_summary.get("mean_relevance", 0.0)),
                "pass_rate": float(run_or_summary.get("pass_rate", 0.0)),
                "p95_latency_ms": float(run_or_summary.get("p95_latency_ms", 0.0)),
            }
            if "mean_context_precision" in run_or_summary:
                out["context_precision"] = float(run_or_summary["mean_context_precision"])
            if "mean_context_recall" in run_or_summary:
                out["context_recall"] = float(run_or_summary["mean_context_recall"])
            return out, str(run_id) if run_id else None

        # Raw flat dict of metrics
        clean: dict[str, float] = {}
        for k, v in run_or_summary.items():
            if isinstance(v, (int, float)):
                clean[k] = float(v)
        return clean, str(run_id) if run_id else None

    raise TypeError(f"Cannot extract metrics from type: {type(run_or_summary).__name__}")


def check_regression(
    current: EvalSummary | RetrievalEvalSummary | EvalRun | dict[str, Any] | str | Path,
    baseline: EvalSummary | RetrievalEvalSummary | EvalRun | dict[str, Any] | str | Path,
    *,
    max_drop: float = 0.05,
    metric_thresholds: dict[str, float] | None = None,
    relative: bool = True,
    direction_overrides: dict[str, MetricDirection] | None = None,
    fail_on_missing_metric: bool = False,
    allow_empty: bool = False,
) -> RegressionReport:
    """Compare a current evaluation run against a baseline run and report metric regressions.

    Parameters
    ----------
    current:
        Current evaluation run (summary object, dict, or path to JSON file).
    baseline:
        Baseline evaluation run to compare against.
    max_drop:
        Default allowed metric degradation (e.g. 0.05 for 5% relative drop or 0.05 absolute).
    metric_thresholds:
        Optional per-metric threshold overrides, e.g. ``{"recall@10": 0.02, "p95_latency_ms": 0.10}``.
    relative:
        Whether thresholds represent relative change (default True) or absolute delta.
    direction_overrides:
        Optional custom directions mapping metric names to ``MetricDirection``.
    fail_on_missing_metric:
        If True, flags a regression when a metric listed in ``metric_thresholds`` is absent.
    allow_empty:
        If True, returns pass when zero common metrics are present. Otherwise fails.

    Returns
    -------
    RegressionReport containing all metric diffs, list of regressions, and pass/fail boolean.
    """
    curr_metrics, curr_id = _extract_metrics(current)
    base_metrics, base_id = _extract_metrics(baseline)

    per_metric = metric_thresholds or {}
    all_keys = set(curr_metrics.keys()) | set(base_metrics.keys())
    diffs: dict[str, MetricDiff] = {}
    regressions: list[MetricDiff] = []

    epsilon = 1e-9

    for key in sorted(all_keys):
        in_curr = key in curr_metrics
        in_base = key in base_metrics

        if not in_curr or not in_base:
            if fail_on_missing_metric and key in per_metric:
                missing_in = "current" if not in_curr else "baseline"
                threshold = per_metric[key]
                diff = MetricDiff(
                    metric=key,
                    baseline=base_metrics.get(key, float("nan")),
                    current=curr_metrics.get(key, float("nan")),
                    delta=float("nan"),
                    relative_change=float("nan"),
                    threshold=threshold,
                    direction=get_metric_direction(key, direction_overrides),
                    is_regression=True,
                    status=f"REGRESSION (missing in {missing_in})",
                )
                diffs[key] = diff
                regressions.append(diff)
            continue

        c_val = curr_metrics[key]
        b_val = base_metrics[key]
        direction = get_metric_direction(key, direction_overrides)
        threshold = per_metric.get(key, max_drop)

        # Handle NaN or Inf in metric values
        if math.isnan(c_val) or math.isnan(b_val):
            diff = MetricDiff(
                metric=key,
                baseline=b_val,
                current=c_val,
                delta=float("nan"),
                relative_change=float("nan"),
                threshold=threshold,
                direction=direction,
                is_regression=True,
                status="REGRESSION (NaN value detected)",
            )
            diffs[key] = diff
            regressions.append(diff)
            continue

        delta = c_val - b_val
        if abs(b_val) > epsilon:
            rel_change = delta / abs(b_val)
        else:
            rel_change = 0.0

        is_regression = False
        if direction == MetricDirection.HIGHER_IS_BETTER:
            # Drop in score is bad
            if relative and abs(b_val) > epsilon:
                allowed_min = b_val * (1.0 - threshold) - epsilon
                is_regression = c_val < allowed_min
            else:
                is_regression = (b_val - c_val) > (threshold + epsilon)

            if is_regression:
                status = "REGRESSION"
            elif c_val > (b_val + epsilon):
                status = "IMPROVED"
            else:
                status = "PASS"

        else:  # MetricDirection.LOWER_IS_BETTER
            # Increase in latency / error is bad
            if relative and abs(b_val) > epsilon:
                allowed_max = b_val * (1.0 + threshold) + epsilon
                is_regression = c_val > allowed_max
            else:
                is_regression = (c_val - b_val) > (threshold + epsilon)

            if is_regression:
                status = "REGRESSION"
            elif c_val < (b_val - epsilon):
                status = "IMPROVED"
            else:
                status = "PASS"

        diff = MetricDiff(
            metric=key,
            baseline=b_val,
            current=c_val,
            delta=delta,
            relative_change=rel_change,
            threshold=threshold,
            direction=direction,
            is_regression=is_regression,
            status=status,
        )
        diffs[key] = diff
        if is_regression:
            regressions.append(diff)

    # If no common metrics were evaluated
    if not diffs:
        passed = allow_empty
    else:
        passed = len(regressions) == 0

    return RegressionReport(
        passed=passed,
        diffs=diffs,
        regressions=regressions,
        baseline_id=base_id,
        current_id=curr_id,
    )


def assert_no_regression(
    current: Any,
    baseline: Any,
    *,
    max_drop: float = 0.05,
    metric_thresholds: dict[str, float] | None = None,
    relative: bool = True,
    direction_overrides: dict[str, MetricDirection] | None = None,
    allow_empty: bool = False,
) -> None:
    """Assert that current eval run does not regress compared to baseline.

    Raises :class:`AssertionError` with a comparison table if any metric regresses.
    """
    report = check_regression(
        current=current,
        baseline=baseline,
        max_drop=max_drop,
        metric_thresholds=metric_thresholds,
        relative=relative,
        direction_overrides=direction_overrides,
        allow_empty=allow_empty,
    )
    if not report.passed:
        raise AssertionError(f"Eval metric regression detected:\n{report.summary_table()}")
