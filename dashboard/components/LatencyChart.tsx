"use client";
import type { Metrics } from "@/lib/types";

const ROWS: [keyof Metrics, string, string][] = [
  ["p50", "p50", "#2f7d5b"],
  ["p95", "p95", "#e8623b"],
  ["p99", "p99", "#b8472a"],
];

export default function LatencyChart({ m }: { m: Metrics }) {
  const max = Math.max(1, m.p99, m.p95, m.p50);
  return (
    <div className="bg-surface border border-line rounded-xl2 px-[18px] py-4">
      <h3 className="m-0 mb-3.5 text-[14px] font-semibold">
        Latency percentiles <span className="font-mono text-[11px] text-faint font-normal">ms</span>
      </h3>
      <div className="flex flex-col gap-3.5 pt-1">
        {ROWS.map(([k, label, color]) => {
          const v = m[k] as number;
          return (
            <div key={label}>
              <div className="flex justify-between text-[12px] mb-1">
                <span className="font-mono" style={{ color }}>{label}</span>
                <span className="font-mono">{Number(v).toFixed(1)} ms</span>
              </div>
              <div className="h-2.5 bg-[#f0ece6] rounded-full">
                <div className="h-full rounded-full" style={{ width: `${(v / max) * 100}%`, background: color }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
