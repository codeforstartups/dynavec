"""Visual charting for RAG evaluation scores (Faithfulness vs Answer Relevance)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import EvalSummary


def plot_eval_summary(
    summary: EvalSummary,
    output_path: str = "eval_scores.png",
    title: str = "RAG Evaluation: Faithfulness vs Answer Relevance",
) -> str:
    """Generate a scatter and distribution chart for batch evaluation results.

    Parameters
    ----------
    summary:
        EvalSummary object containing results from EvalRunner.
    output_path:
        File path to save the generated PNG chart.
    title:
        Chart title.

    Returns
    -------
    str:
        The output path where the chart was saved.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless mode
        import matplotlib.pyplot as plt
    except ImportError as exc:
        from ..exceptions import MissingDependencyError

        raise MissingDependencyError("plot_eval_summary", "matplotlib", "benchmark") from exc

    results = summary.results
    if not results:
        raise ValueError("Cannot plot empty evaluation results.")

    faith_scores = [r.faithfulness.score if r.faithfulness is not None else 0.0 for r in results]
    rel_scores = [
        r.answer_relevance.score if r.answer_relevance is not None else 0.0 for r in results
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#FBFAF8")
    ax.set_facecolor("#FFFFFF")

    # Scatter plot
    ax.scatter(
        faith_scores,
        rel_scores,
        c="#E8623B",
        alpha=0.75,
        s=60,
        edgecolors="#B8472A",
        linewidth=1.2,
        label="Evaluated Queries",
        zorder=3,
    )

    # Reference lines for pass threshold
    threshold = summary.pass_threshold
    ax.axvline(
        threshold, color="#6F6862", linestyle="--", alpha=0.5, label=f"Pass Threshold ({threshold})"
    )
    ax.axhline(threshold, color="#6F6862", linestyle="--", alpha=0.5)

    # Mean score crosshairs
    ax.scatter(
        [summary.mean_faithfulness],
        [summary.mean_relevance],
        c="#2F7D5B",
        s=140,
        marker="X",
        label=f"Mean (F:{summary.mean_faithfulness:.2f}, R:{summary.mean_relevance:.2f})",
        zorder=4,
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(
        "Faithfulness (Groundedness in Context)", fontsize=11, fontweight="bold", color="#14110F"
    )
    ax.set_ylabel(
        "Answer Relevance (Intent Alignment)", fontsize=11, fontweight="bold", color="#14110F"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#14110F")

    # Styling
    ax.grid(True, linestyle=":", alpha=0.6, color="#ECE6DF")
    for spine in ax.spines.values():
        spine.set_color("#ECE6DF")

    ax.legend(loc="lower left", framealpha=0.9, facecolor="#FBFAF8", edgecolor="#ECE6DF")

    # Add subtitle annotations
    stats_text = (
        f"Total Samples: {summary.total_samples}  |  "
        f"Pass Rate: {summary.pass_rate * 100:.1f}%  |  "
        f"p95 Latency: {summary.p95_latency_ms:.1f}ms"
    )
    plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=9, color="#6F6862", family="monospace")

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_retrieval_metrics(
    summary: Any,
    output_path: str = "retrieval_eval.png",
    title: str = "Retrieval Quality Evaluation (Recall / MRR / nDCG)",
) -> str:
    """Generate visual curves and summary charts for retrieval evaluation results.

    Parameters
    ----------
    summary:
        RetrievalEvalSummary object from RetrievalEvalRunner.
    output_path:
        File path to save the generated PNG chart.
    title:
        Chart title.

    Returns
    -------
    str:
        The output path where the chart was saved.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless mode
        import matplotlib.pyplot as plt
    except ImportError as exc:
        from ..exceptions import MissingDependencyError

        raise MissingDependencyError("plot_retrieval_metrics", "matplotlib", "benchmark") from exc

    if not hasattr(summary, "k_values") or not summary.k_values:
        raise ValueError("Cannot plot empty retrieval evaluation summary.")

    k_vals = summary.k_values
    recalls = [summary.mean_recall.get(k, 0.0) for k in k_vals]
    ndcgs = [summary.mean_ndcg.get(k, 0.0) for k in k_vals]
    precisions = [summary.mean_precision.get(k, 0.0) for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor("#FBFAF8")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    # Panel 1: Recall@k & nDCG@k vs k
    ax1.plot(k_vals, recalls, marker="o", color="#E8623B", linewidth=2.2, label="Recall@k")
    ax1.plot(k_vals, ndcgs, marker="s", color="#2F7D5B", linewidth=2.2, label="nDCG@k")
    ax1.plot(
        k_vals,
        precisions,
        marker="^",
        color="#4C78A8",
        linewidth=1.8,
        linestyle="--",
        label="Precision@k",
    )

    ax1.set_xlabel("Rank Cutoff (k)", fontsize=11, fontweight="bold", color="#14110F")
    ax1.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold", color="#14110F")
    ax1.set_title("Ranking Quality vs Cutoff (k)", fontsize=12, fontweight="bold", color="#14110F")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#ECE6DF")
    ax1.legend(loc="lower right", framealpha=0.9, facecolor="#FBFAF8", edgecolor="#ECE6DF")

    # Panel 2: Summary Metrics Bar Chart
    max_k = k_vals[-1]
    metrics_labels = ["MRR", f"Recall@{max_k}", f"nDCG@{max_k}", f"Precision@{max_k}"]
    metrics_vals = [
        summary.mean_mrr,
        summary.mean_recall.get(max_k, 0.0),
        summary.mean_ndcg.get(max_k, 0.0),
        summary.mean_precision.get(max_k, 0.0),
    ]
    colors = ["#F58518", "#E8623B", "#2F7D5B", "#4C78A8"]

    bars = ax2.bar(
        metrics_labels, metrics_vals, color=colors, width=0.55, edgecolor="#14110F", linewidth=0.8
    )
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Mean Score", fontsize=11, fontweight="bold", color="#14110F")
    ax2.set_title("Aggregate Metric Summary", fontsize=12, fontweight="bold", color="#14110F")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6, color="#ECE6DF")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#ECE6DF")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color="#14110F")

    stats_text = (
        f"Total Queries: {summary.total_queries}  |  "
        f"p50 Latency: {summary.p50_latency_ms:.1f}ms  |  "
        f"p95 Latency: {summary.p95_latency_ms:.1f}ms"
    )
    plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=9, color="#6F6862", family="monospace")

    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
