"use client";
import type { TraceEvent } from "@/lib/types";

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <>
      <div className="font-mono text-[12px] text-muted">{k}</div>
      <div className="tabular-nums break-words">{v == null || v === "" ? "—" : v}</div>
    </>
  );
}

export default function TraceDrawer({ trace, onClose }: { trace: TraceEvent | null; onClose: () => void }) {
  return (
    <div
      className={
        "fixed top-0 right-0 h-screen w-[min(460px,92vw)] bg-surface border-l border-line shadow-card z-20 overflow-y-auto p-[22px] transition-transform duration-200 " +
        (trace ? "translate-x-0" : "translate-x-full")
      }
    >
      <button onClick={onClose} className="absolute top-4 right-[18px] text-[20px] text-muted">×</button>
      {trace && (
        <>
          <h3 className="m-0 mb-1 text-[16px] font-semibold">
            <span className="font-mono text-[11px] px-2 py-0.5 rounded-full border border-line mr-2">{trace.op}</span>
            trace
          </h3>
          <div className="font-mono text-[12px] text-muted">
            {trace.id} · {new Date(trace.ts * 1000).toLocaleString()}
          </div>
          <div className="grid grid-cols-[130px_1fr] gap-x-3.5 gap-y-2 mt-4 text-[13px]">
            <Row k="Namespace" v={trace.namespace} />
            <Row k="Latency" v={`${trace.latency_ms.toFixed(2)} ms`} />
            <Row k="Results" v={trace.n_results} />
            <Row k="top_k" v={trace.top_k} />
            <Row k="Cache" v={trace.cache_hit == null ? "no cache" : trace.cache_hit ? "hit" : "miss"} />
            <Row k="Filtered" v={String(trace.filtered)} />
            <Row k="Rescore" v={trace.rescore} />
            <Row k="Rerank" v={trace.rerank} />
            <Row k="Top score" v={trace.score_top} />
            <Row k="Mean score" v={trace.score_mean} />
            <Row k="Status" v={trace.status} />
            {trace.error && <Row k="Error" v={trace.error} />}
            {trace.query_preview && <Row k="Query" v={trace.query_preview} />}
          </div>
          <div className="mt-[18px]">
            <div className="font-mono text-[12px] text-muted mb-1.5">End-to-end latency</div>
            <div className="h-2.5 rounded-full bg-accent" />
          </div>
        </>
      )}
    </div>
  );
}
