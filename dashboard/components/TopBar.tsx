"use client";

const RANGES = [
  { label: "30m", w: 1800 },
  { label: "1h", w: 3600 },
  { label: "6h", w: 21600 },
  { label: "24h", w: 86400 },
];

export default function TopBar({
  window: win, onWindow, auto, onAuto, live,
}: {
  window: number;
  onWindow: (w: number) => void;
  auto: boolean;
  onAuto: () => void;
  live: boolean;
}) {
  return (
    <header className="flex items-center gap-4 px-5 py-3 bg-surface border-b border-line sticky top-0 z-10">
      <span className="flex items-center gap-2.5 font-mono font-bold text-[16px]">
        <svg width="18" height="18" viewBox="0 0 22 22" className="text-accent">
          <rect x="1" y="1" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <line x1="1" y1="11" x2="21" y2="11" stroke="currentColor" strokeWidth="1.5" />
          <line x1="11" y1="1" x2="11" y2="21" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="6" cy="6" r="2" fill="currentColor" />
          <circle cx="16" cy="16" r="2" fill="currentColor" />
        </svg>
        dynavec
      </span>
      <span className="text-muted text-[13px] mr-auto">
        Observability{" "}
        <span className={"font-mono text-[11px] " + (live ? "text-ok" : "text-accent-ink")}>
          · {live ? "live" : "sample data"}
        </span>
      </span>

      <div className="flex border-[1.5px] border-ink rounded-lg overflow-hidden">
        {RANGES.map((r) => (
          <button
            key={r.w}
            onClick={() => onWindow(r.w)}
            className={
              "font-mono text-[12px] px-3 py-1.5 border-r border-line last:border-r-0 " +
              (win === r.w ? "bg-accent text-white" : "bg-surface text-muted hover:text-ink")
            }
          >
            {r.label}
          </button>
        ))}
      </div>
      <button
        onClick={onAuto}
        className={
          "font-mono text-[12px] border-[1.5px] border-ink rounded-lg px-3 py-1.5 " +
          (auto ? "bg-ink text-white" : "bg-surface text-ink")
        }
      >
        Auto-refresh
      </button>
    </header>
  );
}
