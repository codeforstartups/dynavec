"""Tests for the telemetry recorder + aggregation (pure, no AWS)."""

import time

from dynavec.telemetry import TelemetryRecorder, aggregate


def _rec_with(events):
    r = TelemetryRecorder()
    for e in events:
        r.record(e)
    return r


def test_record_and_events_newest_first():
    r = TelemetryRecorder()
    for i in range(3):
        r.record(r.new_event("search", namespace="kb", latency_ms=float(i)))
    evs = r.events()
    assert len(evs) == 3
    assert evs[0].latency_ms == 2.0  # newest first


def test_events_filtering():
    r = TelemetryRecorder()
    r.record(r.new_event("search", namespace="a", status="ok"))
    r.record(r.new_event("upsert", namespace="b", status="ok"))
    r.record(r.new_event("search", namespace="a", status="error"))
    assert len(r.events(op="search")) == 2
    assert len(r.events(namespace="b")) == 1
    assert len(r.events(status="error")) == 1
    assert len(r.events(op="search", status="ok")) == 1


def test_get_and_limit():
    r = TelemetryRecorder()
    ev = r.new_event("search")
    r.record(ev)
    assert r.get(ev.id).id == ev.id
    assert r.get("nope") is None
    for _ in range(10):
        r.record(r.new_event("search"))
    assert len(r.events(limit=5)) == 5


def test_ring_buffer_caps_size():
    r = TelemetryRecorder(max_events=5)
    for _ in range(20):
        r.record(r.new_event("search"))
    assert len(r.snapshot()) == 5


def test_aggregate_percentiles_and_qpm():
    now = 1000.0
    evs = []
    r = TelemetryRecorder()
    for lat in range(1, 101):  # 100 events, latency 1..100ms
        evs.append(r.new_event("search", latency_ms=float(lat)))
    for e in evs:
        e.ts = now - 10  # all within window
    agg = aggregate(evs, window_seconds=3600, now=now)
    assert agg["total"] == 100
    assert 49 <= agg["p50"] <= 52
    assert 94 <= agg["p95"] <= 96
    assert agg["qpm"] == round(100 / 60.0, 2)


def test_aggregate_cache_hit_rate_and_errors():
    now = 500.0
    r = TelemetryRecorder()
    evs = [
        r.new_event("search", latency_ms=5, cache_hit=True),
        r.new_event("search", latency_ms=5, cache_hit=True),
        r.new_event("search", latency_ms=5, cache_hit=False),
        r.new_event("search", latency_ms=5, cache_hit=None),  # no cache configured
        r.new_event("search", latency_ms=5, status="error"),
    ]
    for e in evs:
        e.ts = now - 1
    agg = aggregate(evs, window_seconds=3600, now=now)
    assert agg["cache_total"] == 3          # None excluded
    assert agg["cache_hits"] == 2
    assert agg["cache_hit_rate"] == round(100 * 2 / 3, 1)
    assert agg["error_rate"] == round(100 * 1 / 5, 2)


def test_aggregate_window_excludes_old_events():
    now = 10_000.0
    r = TelemetryRecorder()
    old = r.new_event("search", latency_ms=5)
    old.ts = now - 5000
    new = r.new_event("search", latency_ms=5)
    new.ts = now - 10
    agg = aggregate([old, new], window_seconds=3600, now=now)
    assert agg["total"] == 1


def test_aggregate_histogram_buckets():
    now = 3600.0
    r = TelemetryRecorder()
    evs = []
    # one event in the first bucket, two in the last
    e0 = r.new_event("search", latency_ms=1)
    e0.ts = now - 3599
    e1 = r.new_event("search", latency_ms=1)
    e1.ts = now - 1
    e2 = r.new_event("search", latency_ms=1)
    e2.ts = now - 1
    evs = [e0, e1, e2]
    agg = aggregate(evs, window_seconds=3600, now=now, buckets=30)
    assert len(agg["histogram"]) == 30
    assert sum(agg["histogram"]) == 3
    assert agg["histogram"][0] == 1
    assert agg["histogram"][-1] == 2


def test_empty_aggregate_is_safe():
    agg = aggregate([], window_seconds=3600, now=time.time())
    assert agg["total"] == 0
    assert agg["p95"] == 0.0
    assert agg["cache_hit_rate"] is None
