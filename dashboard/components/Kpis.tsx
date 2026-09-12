import type { Metrics } from "@/lib/types";

function Kpi({ label, value, unit, win }: { label: string; value: string; unit?: string; win?: boolean }) {
  return (
    <div className="bg-surface border border-line rounded-xl2 px-[18px] py-4">
      <div className="text-[12px] text-muted mb-2">{label}</div>
      <div className={"font-mono text-[26px] font-bold tracking-tight " + (win ? "text-accent-ink" : "")}>
        {value}
        {unit && <span className="text-[13px] text-muted font-normal"> {unit}</span>}
      </div>
    </div>
  );
}

export default function Kpis({ m }: { m: Metrics }) {
  const num = (n: number, d = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3.5 mb-5">
      <Kpi label="Queries / min" value={num(m.qpm, 1)} win />
      <Kpi label="p95 latency" value={num(m.p95, 0)} unit="ms" />
      <Kpi label="Cache hit rate" value={m.cache_hit_rate == null ? "—" : num(m.cache_hit_rate, 1)} unit={m.cache_hit_rate == null ? "" : "%"} win />
      <Kpi label="Avg results" value={num(m.avg_results, 1)} />
      <Kpi label="Error rate" value={num(m.error_rate, 1)} unit="%" />
    </div>
  );
}
