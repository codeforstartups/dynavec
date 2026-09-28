"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const GROUPS: { title: string; items: { label: string; soon?: boolean; href: string }[] }[] = [
  { title: "Observability", items: [{ label: "Tracing", href: "/" }, { label: "Latency", href: "#" }, { label: "Cost", href: "#" }] },
  { title: "Evaluation", items: [{ label: "Scores", soon: true, href: "#" }, { label: "Faithfulness", soon: true, href: "#" }] },
  { title: "Resources", items: [{ label: "Buckets & Indexes", href: "/resources" }] },
  { title: "Knowledge Graph", items: [{ label: "Visualizer", href: "/graph" }] },
];

export default function Sidebar() {
  const pathname = usePathname();
  
  return (
    <nav className="w-[210px] shrink-0 border-r border-line bg-surface p-4 hidden md:block">
      {GROUPS.map((g) => (
        <div key={g.title}>
          <h4 className="font-mono text-[11px] uppercase tracking-wider text-faint mt-4 mb-2 px-2">{g.title}</h4>
          {g.items.map((it) => {
            const active = it.href === "/" ? pathname === "/" : pathname?.startsWith(it.href);
            return (
              <Link
                key={it.label}
                href={it.soon ? "#" : it.href}
                className={
                  "flex items-center justify-between px-2.5 py-1.5 rounded-md text-[13.5px] mb-0.5 " +
                  (active
                    ? "bg-accent-soft text-accent-ink font-semibold border-l-2 border-accent"
                    : it.soon
                    ? "text-faint cursor-default"
                    : "text-muted hover:text-ink cursor-pointer")
                }
              >
                {it.label}
                {it.soon && <span className="font-mono text-[10px] text-faint">soon</span>}
              </Link>
            );
          })}

        </div>
      ))}
    </nav>
  );
}
