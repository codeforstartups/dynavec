"""Render benchmark results as a comparison table + charts.

Produces three artifacts from a results dict:
  1. a Markdown comparison table (highlighting the best value per row)
  2. a grouped-bar chart of recall + latency
  3. a cost-vs-scale line chart on a log x-axis (like the reference screenshots)

Recall/latency for competitors are **representative** figures (documented
constants below) — swap in your own measured numbers from ``run_benchmark.py``
before publishing anything. Cost comes from the real ``cost_model``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from .cost_model import Workload, compare  # noqa: E402

# Brand-neutral palette; dynavec highlighted.
_COLORS = {
    "dynavec": "#E8623B",
    "pinecone_serverless": "#4C78A8",
    "opensearch_serverless": "#72B7B2",
    "qdrant_cloud": "#F58518",
    "weaviate_cloud": "#9D755D",
    "milvus_zilliz": "#B279A2",
}

# Representative quality/latency profile (EDIT with measured values).
# recall@10, warm p50/p95/p99 ms for a ~1M x 768 workload.
_PROFILE = {
    "dynavec":               {"recall": 0.90, "p50": 45, "p95": 120, "p99": 180},
    "pinecone_serverless":   {"recall": 0.95, "p50": 30, "p95": 70,  "p99": 110},
    "opensearch_serverless": {"recall": 0.97, "p50": 15, "p95": 40,  "p99": 70},
    "qdrant_cloud":          {"recall": 0.98, "p50": 8,  "p95": 20,  "p99": 35},
    "weaviate_cloud":        {"recall": 0.97, "p50": 10, "p95": 25,  "p99": 45},
    "milvus_zilliz":         {"recall": 0.98, "p50": 7,  "p95": 18,  "p99": 30},
}

_LABELS = {
    "dynavec": "dynavec",
    "pinecone_serverless": "Pinecone",
    "opensearch_serverless": "OpenSearch",
    "qdrant_cloud": "Qdrant",
    "weaviate_cloud": "Weaviate",
    "milvus_zilliz": "Milvus/Zilliz",
}


@dataclass
class ReportInputs:
    vectors: int = 1_000_000
    dim: int = 768
    qpm: int = 1_000_000
    wpm: int = 100_000


def _row_costs(inp: ReportInputs) -> dict[str, float]:
    return compare(Workload(inp.vectors, inp.dim, inp.qpm, inp.wpm))


def markdown_table(inp: ReportInputs) -> str:
    costs = _row_costs(inp)
    products = list(_PROFILE.keys())
    rows = [
        ("Recall@10", {p: _PROFILE[p]["recall"] for p in products}, "max", "{:.2f}"),
        ("Latency p50 (ms)", {p: _PROFILE[p]["p50"] for p in products}, "min", "{:.0f}"),
        ("Latency p95 (ms)", {p: _PROFILE[p]["p95"] for p in products}, "min", "{:.0f}"),
        ("Cost ($/mo)", {p: costs[p] for p in products}, "min", "${:,.0f}"),
        ("Serverless (scale-to-zero)", {
            "dynavec": "Yes", "pinecone_serverless": "Yes",
            "opensearch_serverless": "No (OCU floor)", "qdrant_cloud": "No (nodes)",
            "weaviate_cloud": "No (nodes)", "milvus_zilliz": "No (CU)",
        }, None, "{}"),
        ("Data in your AWS account", {
            "dynavec": "Yes", "pinecone_serverless": "No",
            "opensearch_serverless": "Yes", "qdrant_cloud": "Self-host only",
            "weaviate_cloud": "Self-host only", "milvus_zilliz": "Self-host only",
        }, None, "{}"),
    ]

    header = "| Metric | " + " | ".join(_LABELS[p] for p in products) + " |"
    sep = "|" + "---|" * (len(products) + 1)
    lines = [header, sep]
    for name, values, best, fmt in rows:
        best_p = None
        if best in ("min", "max") and all(isinstance(v, (int, float)) for v in values.values()):
            best_p = (min if best == "min" else max)(values, key=values.get)
        cells = []
        for p in products:
            v = values[p]
            s = fmt.format(v)
            if p == best_p:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bar_chart(inp: ReportInputs, out_path: str) -> str:
    products = list(_PROFILE.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # recall
    ax1.bar(
        [_LABELS[p] for p in products],
        [_PROFILE[p]["recall"] for p in products],
        color=[_COLORS[p] for p in products],
    )
    ax1.set_title("Recall@10 (representative)")
    ax1.set_ylim(0.85, 1.0)
    ax1.tick_params(axis="x", rotation=30)

    # latency p50/p95/p99 grouped
    import numpy as np

    x = np.arange(len(products))
    w = 0.27
    for i, pct in enumerate(("p50", "p95", "p99")):
        ax2.bar(x + (i - 1) * w, [_PROFILE[p][pct] for p in products], w, label=pct)
    ax2.set_xticks(x)
    ax2.set_xticklabels([_LABELS[p] for p in products], rotation=30)
    ax2.set_title("Query latency (ms, warm) — lower is better")
    ax2.legend()

    fig.suptitle("dynavec vs vector databases — quality & latency", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cost_scale_chart(inp: ReportInputs, out_path: str) -> str:
    scales = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    products = list(_PROFILE.keys())
    series = {p: [] for p in products}
    for n in scales:
        costs = compare(Workload(n, inp.dim, inp.qpm, inp.wpm))
        for p in products:
            series[p].append(costs[p])

    fig, ax = plt.subplots(figsize=(11, 6))
    for p in products:
        ax.plot(
            scales, series[p], marker="o",
            color=_COLORS[p], linewidth=2.5 if p == "dynavec" else 1.6,
            label=_LABELS[p],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of vectors (log scale)")
    ax.set_ylabel("Estimated cost ($/month, log scale)")
    ax.set_title(f"Monthly cost by scale — {inp.dim}d, {inp.qpm:,} queries/mo",
                 fontsize=14, weight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.text(0.5, -0.02,
             "APPROX public list prices (see cost_model.py). Order-of-magnitude only.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate(out_dir: str = "benchmarks/out", inp: ReportInputs | None = None) -> dict:
    inp = inp or ReportInputs()
    os.makedirs(out_dir, exist_ok=True)
    table_md = markdown_table(inp)
    table_path = os.path.join(out_dir, "comparison.md")
    with open(table_path, "w") as f:
        f.write(f"# dynavec benchmark comparison\n\n"
                f"_Workload: {inp.vectors:,} × {inp.dim}d, {inp.qpm:,} queries/mo._\n\n")
        f.write(table_md + "\n")
    bars = bar_chart(inp, os.path.join(out_dir, "quality_latency.png"))
    cost = cost_scale_chart(inp, os.path.join(out_dir, "cost_by_scale.png"))
    return {"table_md": table_path, "bar_chart": bars, "cost_chart": cost}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate dynavec benchmark report")
    p.add_argument("--out", default="benchmarks/out")
    p.add_argument("--vectors", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--qpm", type=int, default=1_000_000)
    args = p.parse_args()
    paths = generate(args.out, ReportInputs(vectors=args.vectors, dim=args.dim, qpm=args.qpm))
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\n" + markdown_table(ReportInputs(vectors=args.vectors, dim=args.dim, qpm=args.qpm)))


if __name__ == "__main__":
    main()
