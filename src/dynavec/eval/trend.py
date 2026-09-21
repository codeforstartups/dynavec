"""Trend tracking for eval runs — records, filters, and compares eval summaries over time.

Usage::

    store = EvalRunStore()

    # after each eval run, record it with tags
    store.record(summary, tags={"dataset": "squad", "model": "text-embedding-3-small"})

    # trend recall@10 over time
    series = store.trend("recall", k=10)   # [(timestamp, value), ...]

    # filter to a specific model
    runs = store.filter(model="text-embedding-3-small")

    # compare two runs side by side
    diff = store.compare([run_a_id, run_b_id])
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .retrieval import RetrievalEvalSummary
from .runner import EvalSummary


@dataclass
class EvalRun:
    """A recorded eval run with metadata tags and metric snapshots."""

    run_id: str
    ts: float                          # epoch seconds when the run was recorded
    tags: dict[str, str]               # e.g. {"dataset": "squad", "model": "ada-002"}
    metrics: dict[str, Any]            # flat metric snapshot (recall@k, ndcg@k, faithfulness, …)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ts": self.ts,
            "tags": self.tags,
            "metrics": self.metrics,
        }


def _metrics_from_retrieval(summary: RetrievalEvalSummary) -> dict[str, Any]:
    """Flatten a RetrievalEvalSummary into a key→value metric dict."""
    out: dict[str, Any] = {
        "total_queries": summary.total_queries,
        "mrr": summary.mean_mrr,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
    }
    for k, v in summary.mean_recall.items():
        out[f"recall@{k}"] = v
    for k, v in summary.mean_ndcg.items():
        out[f"ndcg@{k}"] = v
    for k, v in summary.mean_precision.items():
        out[f"precision@{k}"] = v
    return out


def _metrics_from_rag(summary: EvalSummary) -> dict[str, Any]:
    """Flatten an EvalSummary (RAG) into a key→value metric dict."""
    return {
        "total_samples": summary.total_samples,
        "faithfulness": summary.mean_faithfulness,
        "relevance": summary.mean_relevance,
        "pass_rate": summary.pass_rate,
        "p95_latency_ms": summary.p95_latency_ms,
    }


class EvalRunStore:
    """In-memory store for eval runs that supports trend queries and tag filtering.

    Parameters
    ----------
    max_runs:
        Maximum number of runs to keep. Oldest are dropped when full.
    """

    def __init__(self, max_runs: int = 1_000) -> None:
        self._runs: list[EvalRun] = []
        self._max_runs = max_runs

    # ------------------------------------------------------------------ record

    def record(
        self,
        summary: RetrievalEvalSummary | EvalSummary,
        tags: dict[str, str] | None = None,
        ts: float | None = None,
    ) -> EvalRun:
        """Record an eval run and return the stored :class:`EvalRun`.

        Parameters
        ----------
        summary:
            A ``RetrievalEvalSummary`` (from IR eval) or ``EvalSummary`` (from RAG eval).
        tags:
            Free-form key/value labels, e.g. ``{"dataset": "squad", "model": "ada-002"}``.
        ts:
            Epoch timestamp of the run. Defaults to now.
        """
        if isinstance(summary, RetrievalEvalSummary):
            metrics = _metrics_from_retrieval(summary)
        elif isinstance(summary, EvalSummary):
            metrics = _metrics_from_rag(summary)
        else:
            raise TypeError(f"Unsupported summary type: {type(summary)}")

        run = EvalRun(
            run_id=uuid.uuid4().hex[:12],
            ts=ts if ts is not None else time.time(),
            tags=dict(tags or {}),
            metrics=metrics,
        )
        self._runs.append(run)
        if len(self._runs) > self._max_runs:
            self._runs = self._runs[-self._max_runs:]
        return run

    # ------------------------------------------------------------------ filter

    def filter(self, **tag_filters: str) -> list[EvalRun]:
        """Return runs whose tags match all supplied key=value pairs.

        Example::

            store.filter(dataset="squad", model="ada-002")
        """
        out = []
        for run in self._runs:
            if all(run.tags.get(k) == v for k, v in tag_filters.items()):
                out.append(run)
        return out

    def all_runs(self) -> list[EvalRun]:
        """Return all recorded runs, oldest first."""
        return list(self._runs)

    # ------------------------------------------------------------------- trend

    def trend(
        self,
        metric: str,
        k: int | None = None,
        **tag_filters: str,
    ) -> list[tuple[float, float]]:
        """Return a time series ``[(timestamp, value)]`` for the named metric.

        Parameters
        ----------
        metric:
            Metric name, e.g. ``"recall"``, ``"ndcg"``, ``"faithfulness"``, ``"mrr"``.
            For k-valued metrics (recall, ndcg, precision) pass ``k`` as well.
        k:
            Rank cutoff for recall/ndcg/precision metrics (e.g. ``k=10``).
        tag_filters:
            Optional tag constraints — same as :meth:`filter`.

        Returns
        -------
        list of ``(ts, value)`` tuples, sorted oldest first. Runs where the metric
        is absent are skipped.
        """
        key = f"{metric}@{k}" if k is not None else metric
        runs = self.filter(**tag_filters) if tag_filters else self._runs
        series = []
        for run in runs:
            val = run.metrics.get(key)
            if val is not None:
                series.append((run.ts, float(val)))
        return sorted(series, key=lambda t: t[0])

    # ------------------------------------------------------------------ compare

    def compare(self, run_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return a side-by-side comparison of the requested runs.

        Returns a dict keyed by ``run_id``, each value being the run's full
        ``to_dict()`` representation so callers can diff any field.

        Example::

            diff = store.compare(["abc123", "def456"])
            # diff["abc123"]["metrics"]["recall@10"] vs diff["def456"]["metrics"]["recall@10"]
        """
        id_set = set(run_ids)
        result: dict[str, dict[str, Any]] = {}
        for run in self._runs:
            if run.run_id in id_set:
                result[run.run_id] = run.to_dict()
        missing = id_set - set(result)
        if missing:
            raise KeyError(f"Run IDs not found: {missing}")
        return result
