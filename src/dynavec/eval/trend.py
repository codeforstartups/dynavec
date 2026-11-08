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

    # persist and restore across runs
    store.save_json("eval_history.json")
    loaded = EvalRunStore.load_json("eval_history.json")
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRun:
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for EvalRun, got {type(data).__name__}")
        run_id = str(data.get("run_id", ""))
        ts = float(data.get("ts", 0.0))
        tags = {str(k): str(v) for k, v in data.get("tags", {}).items()}
        metrics = dict(data.get("metrics", {}))
        return cls(run_id=run_id, ts=ts, tags=tags, metrics=metrics)


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
    out: dict[str, Any] = {
        "total_samples": summary.total_samples,
        "faithfulness": summary.mean_faithfulness,
        "relevance": summary.mean_relevance,
        "pass_rate": summary.pass_rate,
        "p95_latency_ms": summary.p95_latency_ms,
    }
    if hasattr(summary, "mean_context_precision") and summary.mean_context_precision is not None:
        out["context_precision"] = summary.mean_context_precision
    if hasattr(summary, "mean_context_recall") and summary.mean_context_recall is not None:
        out["context_recall"] = summary.mean_context_recall
    return out


class EvalRunStore:
    """In-memory store for eval runs that supports trend queries, tag filtering, and JSON persistence.

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

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store to a dictionary representation."""
        return {
            "max_runs": self._max_runs,
            "runs": [r.to_dict() for r in self._runs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunStore:
        """Construct an EvalRunStore from a serialized dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for EvalRunStore, got {type(data).__name__}")
        max_runs = int(data.get("max_runs", 1_000))
        store = cls(max_runs=max_runs)
        raw_runs = data.get("runs", [])
        if isinstance(raw_runs, list):
            store._runs = [EvalRun.from_dict(r) for r in raw_runs if isinstance(r, dict)]
        return store

    def save_json(self, path: str | Path) -> None:
        """Persist recorded runs atomically to a JSON file."""
        file_path = Path(path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(self.to_dict(), indent=2)
        # Write to temporary file in the same directory, then atomic rename
        with tempfile.NamedTemporaryFile("w", dir=file_path.parent, delete=False, encoding="utf-8") as tf:
            tf.write(payload)
            tmp_name = tf.name

        try:
            os.replace(tmp_name, file_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    @classmethod
    def load_json(cls, path: str | Path) -> EvalRunStore:
        """Load runs from a JSON file.

        If the file does not exist, returns an empty EvalRunStore.
        Supports both `{max_runs: ..., runs: [...]}` and a bare list `[{...}, ...]`.
        """
        file_path = Path(path).resolve()
        if not file_path.exists():
            return cls()

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            store = cls()
            store._runs = [EvalRun.from_dict(r) for r in data if isinstance(r, dict)]
            return store

        return cls.from_dict(data)
