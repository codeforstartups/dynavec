"use client";
import type { TraceEvent, TraceFilters } from "@/lib/types";

const OP_CLASS: Record<string, string> = {
  search: "bg-[#eef3ff] text-[#3b5bdb] border-[#dbe3ff]",
  graph_search: "bg-[#f3eeff] text-[#7048e8] border-[#e5dbff]",
  upsert: "bg-[#eafaf1] text-ok border-[#d3f0e0]",
};

export default function TracesTable({
  traces, filters, onFilter, onSelect,
}: {
  traces: TraceEvent[];
  filters: TraceFilters;
  onFilter: (f: TraceFilters) => void;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="bg-surface border border-line rounded-xl2 overflow-hidden">
      <div className="flex items-center gap-3 px-[18px] py-3.5 border-b border-line">
        <h3 className="m-0 text-[14px] font-semibold">Traces</h3>
        <div className="ml-auto flex gap-2">
          <select
            value={filters.op || ""}
            onChange={(e) => onFilter({ ...filters, op: e.target.value })}
            className="font-mono text-[12px] border border-line rounded-md px-2.5 py-1.5 bg-bg"
          >
            <option value="">all ops</option>
            <option value="search">search</option>
            <option value="graph_search">graph_search</option>
            <option value="upsert">upsert</option>
          </select>
          <select
            value={filters.status || ""}
            onChange={(e) => onFilter({ ...filters, status: e.target.value })}
            className="font-mono text-[12px] border border-line rounded-md px-2.5 py-1.5 bg-bg"
          >
            <option value="">any status</option>
            <option value="ok">ok</option>
            <option value="error">error</option>
          </select>
          <input
            placeholder="namespace…"
            value={filters.namespace || ""}
            onChange={(e) => onFilter({ ...filters, namespace: e.target.value })}
            className="font-mono text-[12px] border border-line rounded-md px-2.5 py-1.5 bg-bg w-28"
          />
        </div>
      </div>

      {traces.length === 0 ? (
        <div className="p-10 text-center text-muted">No traces match.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="text-left font-mono text-[11px] uppercase tracking-wide text-muted bg-[#faf6f1]">
                {["Start", "Op", "Namespace", "Latency", "Results", "Cache", "Rank", "Status"].map((h) => (
                  <th key={h} className="px-[18px] py-2.5 border-b border-line font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traces.map((e) => (
                <tr
                  key={e.id}
                  onClick={() => onSelect(e.id)}
                  className="cursor-pointer hover:bg-accent-soft [&>td]:px-[18px] [&>td]:py-2.5 [&>td]:border-b [&>td]:border-line tabular-nums"
                >
                  <td className="font-mono">{new Date(e.ts * 1000).toLocaleTimeString()}</td>
                  <td>
                    <span className={"font-mono text-[11px] px-2 py-0.5 rounded-full border " + (OP_CLASS[e.op] || "border-line")}>
                      {e.op}
                    </span>
                  </td>
                  <td>{e.namespace}</td>
                  <td className="font-mono">{e.latency_ms.toFixed(1)} ms</td>
                  <td className="font-mono">{e.n_results}</td>
                  <td>
                    {e.cache_hit == null ? <span className="text-faint">—</span> : e.cache_hit ? <span className="text-ok">hit</span> : <span className="text-muted">miss</span>}
                  </td>
                  <td className="font-mono">{[e.rescore, e.rerank].filter(Boolean).join("+") || <span className="text-faint">—</span>}</td>
                  <td className={e.status === "error" ? "text-err font-semibold" : "text-ok"}>{e.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
