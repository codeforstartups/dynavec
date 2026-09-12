"use client";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import type { Metrics } from "@/lib/types";

export default function VolumeChart({ m }: { m: Metrics }) {
  const data = m.histogram.map((count, i) => ({ i, count }));
  return (
    <div className="bg-surface border border-line rounded-xl2 px-[18px] py-4">
      <h3 className="m-0 mb-3.5 text-[14px] font-semibold">
        Query volume{" "}
        <span className="font-mono text-[11px] text-faint font-normal">
          {m.total} traces · {Math.round(m.bucket_width_s)}s buckets
        </span>
      </h3>
      <div style={{ width: "100%", height: 150 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <XAxis dataKey="i" hide />
            <Tooltip
              cursor={{ fill: "#fdeee8" }}
              contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 12, border: "1px solid #ece6df", borderRadius: 8 }}
              labelFormatter={() => ""}
              formatter={(v: number) => [v, "queries"]}
            />
            <Bar dataKey="count" fill="#e8623b" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
