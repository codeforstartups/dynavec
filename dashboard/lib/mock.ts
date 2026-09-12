import type { Metrics, TraceEvent } from "./types";

// Deterministic-ish sample data so the dashboard renders standalone (dev, static
// export, first run) before it is pointed at a live dynavec telemetry API.
const NS = ["kb", "tenant-a", "tenant-b"];
const OPS = ["search", "search", "search", "graph_search", "upsert"];
const RANK = [null, null, "cosine", "mmr", "dot"];

let seed = 42;
function rand() {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function pick<T>(a: T[]): T {
  return a[Math.floor(rand() * a.length)];
}

export function mockTraces(n = 60): TraceEvent[] {
  const now = Date.now() / 1000;
  const out: TraceEvent[] = [];
  for (let i = 0; i < n; i++) {
    const op = pick(OPS);
    const cacheHit = op === "upsert" ? null : rand() > 0.35;
    const lat = op === "upsert" ? 8 + rand() * 30 : (cacheHit ? 6 + rand() * 12 : 60 + rand() * 90);
    const nres = op === "upsert" ? 0 : pick([5, 10, 10, 20]);
    out.push({
      id: Math.random().toString(16).slice(2, 14),
      ts: now - i * 3,
      op,
      namespace: pick(NS),
      latency_ms: Math.round(lat * 100) / 100,
      n_results: nres,
      top_k: op === "upsert" ? null : nres,
      cache_hit: cacheHit,
      filtered: rand() > 0.6,
      rescore: op === "search" ? pick(RANK) : null,
      rerank: rand() > 0.7 ? "mmr" : null,
      score_top: op === "upsert" ? null : Math.round((0.7 + rand() * 0.29) * 1000) / 1000,
      score_mean: op === "upsert" ? null : Math.round((0.5 + rand() * 0.3) * 1000) / 1000,
      status: rand() > 0.98 ? "error" : "ok",
      error: null,
      query_preview: op === "upsert" ? null : pick(["apple pie recipe", "rocket to mars", "serverless vectors on aws", "embedding models"]),
    });
  }
  return out;
}

export function mockMetrics(traces: TraceEvent[]): Metrics {
  const win = traces;
  const lat = win.filter((e) => e.status === "ok").map((e) => e.latency_ms).sort((a, b) => a - b);
  const pct = (p: number) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor((lat.length - 1) * p))] : 0);
  const cacheScoped = win.filter((e) => e.cache_hit !== null);
  const hits = cacheScoped.filter((e) => e.cache_hit).length;
  const buckets = 30;
  const hist = new Array(buckets).fill(0);
  win.forEach((_, i) => { hist[Math.min(buckets - 1, Math.floor((i / win.length) * buckets))]++; });
  const op_mix: Record<string, number> = {};
  win.forEach((e) => { op_mix[e.op] = (op_mix[e.op] || 0) + 1; });
  return {
    total: win.length,
    qpm: Math.round((win.length / 60) * 100) / 100 * 6,
    p50: Math.round(pct(0.5) * 100) / 100,
    p95: Math.round(pct(0.95) * 100) / 100,
    p99: Math.round(pct(0.99) * 100) / 100,
    cache_hit_rate: cacheScoped.length ? Math.round((1000 * hits) / cacheScoped.length) / 10 : null,
    cache_hits: hits,
    cache_total: cacheScoped.length,
    error_rate: Math.round((1000 * win.filter((e) => e.status === "error").length) / Math.max(1, win.length)) / 10,
    avg_results: Math.round((10 * win.reduce((s, e) => s + e.n_results, 0)) / Math.max(1, win.length)) / 10,
    op_mix,
    namespaces: {},
    histogram: hist,
    bucket_width_s: 120,
    window_start: 0,
    window_seconds: 3600,
  };
}
