"""Visual charting for RAG evaluation scores (Faithfulness vs Answer Relevance)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    faith_scores = [
        r.faithfulness.score if r.faithfulness is not None else 0.0
        for r in results
    ]
    rel_scores = [
        r.answer_relevance.score if r.answer_relevance is not None else 0.0
        for r in results
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
    ax.axvline(threshold, color="#6F6862", linestyle="--", alpha=0.5, label=f"Pass Threshold ({threshold})")
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
    ax.set_xlabel("Faithfulness (Groundedness in Context)", fontsize=11, fontweight="bold", color="#14110F")
    ax.set_ylabel("Answer Relevance (Intent Alignment)", fontsize=11, fontweight="bold", color="#14110F")
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
