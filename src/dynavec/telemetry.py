"""Telemetry capture for the observability dashboard.

This records **real** events from actual dynavec operations — every ``search``
is timed and logged with its namespace, latency, result count, cache outcome,
and score stats. There is no simulated traffic: if the dashboard shows numbers,
they came from real calls into the client.

Design
------
* :class:`TelemetryEvent` — one recorded operation (a "trace").
* :class:`TelemetryRecorder` — thread-safe ring buffer + aggregation. Attach it
  to a client (``Dynavec(..., telemetry=recorder)``) and it fills as you query.
* :func:`aggregate` — percentiles, QPM, cache-hit-rate, op mix, error rate,
  per-minute histogram — computed from the real events.

Zero third-party deps; safe to import with just the base install.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TelemetryEvent:
    """A single recorded operation."""

    id: str
    ts: float  # epoch seconds (start)
    op: str  # "search" | "upsert" | "graph_search" | ...
    namespace: str = "default"
    latency_ms: float = 0.0
    n_results: int = 0
    top_k: int | None = None
    cache_hit: bool | None = None  # None = no cache configured
    filtered: bool = False
    rescore: str | None = None
    rerank: str | None = None
    score_top: float | None = None
    score_mean: float | None = None
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    query_preview: str | None = None  # only set when capture_text=True
    eval_faithfulness: float | None = None  # 0.0-1.0 when LLM judge ran
    eval_relevance: float | None = None  # 0.0-1.0 when LLM judge ran

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryRecorder:
    """Thread-safe, bounded, in-process store of :class:`TelemetryEvent`.

    Parameters
    ----------
    max_events:
        Ring-buffer capacity (older events drop off).
    capture_text:
        If True, store a short preview of the query text. Off by default so no
        raw content is retained unless the operator opts in.
    """

    def __init__(self, max_events: int = 10_000, capture_text: bool = False) -> None:
        self._events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self.capture_text = capture_text

    # ------------------------------------------------------------------ record
    def record(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)

    def new_event(self, op: str, **kw: Any) -> TelemetryEvent:
        return TelemetryEvent(id=uuid.uuid4().hex[:12], ts=time.time(), op=op, **kw)

    # ------------------------------------------------------------------- reads
    def events(
        self,
        limit: int = 200,
        op: str | None = None,
        namespace: str | None = None,
        status: str | None = None,
        since: float | None = None,
    ) -> list[TelemetryEvent]:
        with self._lock:
            items = list(self._events)
        out = []
        for e in reversed(items):  # newest first
            if op and e.op != op:
                continue
            if namespace and e.namespace != namespace:
                continue
            if status and e.status != status:
                continue
            if since and e.ts < since:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out

    def get(self, event_id: str) -> TelemetryEvent | None:
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    return e
        return None

    def snapshot(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate(
    events: list[TelemetryEvent],
    window_seconds: int = 3600,
    now: float | None = None,
    buckets: int = 30,
) -> dict[str, Any]:
    """Summarize events over the trailing ``window_seconds`` into dashboard stats."""
    now = now if now is not None else time.time()
    start = now - window_seconds
    win = [e for e in events if e.ts >= start]

    latencies = sorted(e.latency_ms for e in win if e.status == "ok")
    cache_scoped = [e for e in win if e.cache_hit is not None]
    hits = sum(1 for e in cache_scoped if e.cache_hit)
    errors = sum(1 for e in win if e.status == "error")

    faith_scores = [e.eval_faithfulness for e in win if e.eval_faithfulness is not None]
    rel_scores = [e.eval_relevance for e in win if e.eval_relevance is not None]

    op_mix: dict[str, int] = {}
    ns_mix: dict[str, int] = {}
    for e in win:
        op_mix[e.op] = op_mix.get(e.op, 0) + 1
        ns_mix[e.namespace] = ns_mix.get(e.namespace, 0) + 1

    # per-bucket count histogram across the window
    bucket_width = max(1.0, window_seconds / buckets)
    hist = [0] * buckets
    for e in win:
        idx = int((e.ts - start) / bucket_width)
        if 0 <= idx < buckets:
            hist[idx] += 1

    minutes = max(1e-9, window_seconds / 60.0)
    return {
        "total": len(win),
        "qpm": round(len(win) / minutes, 2),
        "p50": round(_percentile(latencies, 50), 2),
        "p95": round(_percentile(latencies, 95), 2),
        "p99": round(_percentile(latencies, 99), 2),
        "cache_hit_rate": round(100.0 * hits / len(cache_scoped), 1) if cache_scoped else None,
        "cache_hits": hits,
        "cache_total": len(cache_scoped),
        "error_rate": round(100.0 * errors / len(win), 2) if win else 0.0,
        "avg_results": round(sum(e.n_results for e in win) / len(win), 1) if win else 0.0,
        "eval_faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4)
        if faith_scores
        else None,
        "eval_relevance_mean": round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
        "eval_count": max(len(faith_scores), len(rel_scores)),
        "op_mix": op_mix,
        "namespaces": ns_mix,
        "histogram": hist,
        "bucket_width_s": bucket_width,
        "window_start": start,
        "window_seconds": window_seconds,
    }


def aggregate_eval(
    events: list[TelemetryEvent],
    window_seconds: int = 86400,
    now: float | None = None,
) -> dict[str, Any]:
    """Summarize evaluation scores across recorded telemetry events."""
    now = now if now is not None else time.time()
    start = now - window_seconds
    win = [e for e in events if e.ts >= start]

    eval_events = [
        e for e in win if e.eval_faithfulness is not None or e.eval_relevance is not None
    ]
    faith_scores = [e.eval_faithfulness for e in eval_events if e.eval_faithfulness is not None]
    rel_scores = [e.eval_relevance for e in eval_events if e.eval_relevance is not None]

    pass_threshold = 0.7
    passed = [
        e
        for e in eval_events
        if (e.eval_faithfulness is None or e.eval_faithfulness >= pass_threshold)
        and (e.eval_relevance is None or e.eval_relevance >= pass_threshold)
    ]

    return {
        "total_evals": len(eval_events),
        "mean_faithfulness": round(sum(faith_scores) / len(faith_scores), 4)
        if faith_scores
        else None,
        "mean_relevance": round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
        "pass_rate": round(len(passed) / len(eval_events), 4) if eval_events else None,
        "pass_threshold": pass_threshold,
        "window_seconds": window_seconds,
    }
