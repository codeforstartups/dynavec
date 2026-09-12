const GROUPS: { title: string; items: { label: string; soon?: boolean; active?: boolean }[] }[] = [
  { title: "Observability", items: [{ label: "Tracing", active: true }, { label: "Latency" }, { label: "Cost" }] },
  { title: "Evaluation", items: [{ label: "Scores", soon: true }, { label: "Faithfulness", soon: true }] },
  { title: "Resources", items: [{ label: "Buckets & Indexes", soon: true }, { label: "Namespaces", soon: true }] },
];

export default function Sidebar() {
  return (
    <nav className="w-[210px] shrink-0 border-r border-line bg-surface p-4 hidden md:block">
      {GROUPS.map((g) => (
        <div key={g.title}>
          <h4 className="font-mono text-[11px] uppercase tracking-wider text-faint mt-4 mb-2 px-2">{g.title}</h4>
          {g.items.map((it) => (
            <a
              key={it.label}
              className={
                "flex items-center justify-between px-2.5 py-1.5 rounded-md text-[13.5px] mb-0.5 " +
                (it.active
                  ? "bg-accent-soft text-accent-ink font-semibold border-l-2 border-accent"
                  : it.soon
                  ? "text-faint cursor-default"
                  : "text-muted hover:text-ink cursor-pointer")
              }
            >
              {it.label}
              {it.soon && <span className="font-mono text-[10px] text-faint">soon</span>}
            </a>
          ))}
        </div>
      ))}
    </nav>
  );
}
