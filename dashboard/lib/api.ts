import type { Metrics, TraceEvent, TraceFilters } from "./types";
import { mockMetrics, mockTraces } from "./mock";

// Point this at a running `dynavec.dashboard.serve(recorder)` API.
// Falls back to sample data when unset or unreachable (dev / static preview).
const API_BASE = process.env.NEXT_PUBLIC_DYNAVEC_API || "";

let _mockTraces: TraceEvent[] | null = null;
function sampleTraces(): TraceEvent[] {
  if (!_mockTraces) _mockTraces = mockTraces(60);
  return _mockTraces;
}

export function isLive(): boolean {
  return Boolean(API_BASE);
}

export async function getMetrics(windowSeconds: number): Promise<Metrics> {
  if (API_BASE) {
    try {
      const r = await fetch(`${API_BASE}/api/metrics?window=${windowSeconds}`, { cache: "no-store" });
      if (r.ok) return (await r.json()) as Metrics;
    } catch { /* fall through to sample */ }
  }
  return mockMetrics(sampleTraces());
}

export async function getTraces(filters: TraceFilters, limit = 100): Promise<TraceEvent[]> {
  if (API_BASE) {
    try {
      const q = new URLSearchParams({ limit: String(limit) });
      if (filters.op) q.set("op", filters.op);
      if (filters.status) q.set("status", filters.status);
      if (filters.namespace) q.set("namespace", filters.namespace);
      const r = await fetch(`${API_BASE}/api/traces?${q}`, { cache: "no-store" });
      if (r.ok) return (await r.json()) as TraceEvent[];
    } catch { /* fall through */ }
  }
  return sampleTraces().filter((e) =>
    (!filters.op || e.op === filters.op) &&
    (!filters.status || e.status === filters.status) &&
    (!filters.namespace || e.namespace.includes(filters.namespace)),
  );
}

export async function getTrace(id: string): Promise<TraceEvent | null> {
  if (API_BASE) {
    try {
      const r = await fetch(`${API_BASE}/api/trace/${id}`, { cache: "no-store" });
      if (r.ok) return (await r.json()) as TraceEvent;
    } catch { /* fall through */ }
  }
  return sampleTraces().find((e) => e.id === id) || null;
}
