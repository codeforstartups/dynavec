"""Batch dataset evaluation runner for RAG quality benchmarking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .base import BaseJudge, RAGEvalResult
from .metrics import evaluate_rag


@dataclass
class EvalSummary:
    """Aggregated summary of a batch evaluation run."""

    total_samples: int
    mean_faithfulness: float
    mean_relevance: float
    pass_rate: float
    pass_threshold: float
    p95_latency_ms: float
    results: list[RAGEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data


class EvalRunner:
    """Executes evaluation over benchmark datasets and computes summary metrics.

    Parameters
    ----------
    judge:
        The pluggable LLM judge to use for evaluations.
    pass_threshold:
        Score threshold (default 0.7) for counting a sample as passing.
    """

    def __init__(self, judge: BaseJudge, pass_threshold: float = 0.7) -> None:
        self.judge = judge
        self.pass_threshold = pass_threshold

    def evaluate_sample(
        self,
        query: str,
        context: str | Sequence[str],
        answer: str,
        run_faithfulness: bool = True,
        run_answer_relevance: bool = True,
    ) -> RAGEvalResult:
        """Evaluate a single query-context-answer triplet."""
        return evaluate_rag(
            query=query,
            context=context,
            answer=answer,
            judge=self.judge,
            run_faithfulness=run_faithfulness,
            run_answer_relevance=run_answer_relevance,
        )

    def run(
        self,
        dataset: Sequence[dict[str, Any] | tuple[str, str | Sequence[str], str]],
        run_faithfulness: bool = True,
        run_answer_relevance: bool = True,
    ) -> EvalSummary:
        """Evaluate an entire dataset in batch and compute aggregate quality metrics.

        Each item in `dataset` can be:
        - A dict with keys ``"query"``, ``"context"``, ``"answer"``
        - A 3-tuple ``(query, context, answer)``
        """
        results: list[RAGEvalResult] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        latencies: list[float] = []
        passed_count = 0

        for item in dataset:
            if isinstance(item, dict):
                q = str(item.get("query", ""))
                c = item.get("context", [])
                a = str(item.get("answer", ""))
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                q, c, a = item[0], item[1], item[2]
            else:
                continue

            res = self.evaluate_sample(
                query=q,
                context=c,
                answer=a,
                run_faithfulness=run_faithfulness,
                run_answer_relevance=run_answer_relevance,
            )
            results.append(res)
            latencies.append(res.latency_ms)

            is_pass = True
            if res.faithfulness is not None:
                faithfulness_scores.append(res.faithfulness.score)
                if res.faithfulness.score < self.pass_threshold:
                    is_pass = False
            if res.answer_relevance is not None:
                relevance_scores.append(res.answer_relevance.score)
                if res.answer_relevance.score < self.pass_threshold:
                    is_pass = False

            if is_pass and (res.faithfulness is not None or res.answer_relevance is not None):
                passed_count += 1

        total = len(results)
        mean_f = (
            float(sum(faithfulness_scores) / len(faithfulness_scores))
            if faithfulness_scores
            else 0.0
        )
        mean_r = float(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 0.0
        pass_rate = float(passed_count / total) if total > 0 else 0.0

        # Calculate p95 latency
        p95_lat = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            idx = int(0.95 * (len(sorted_lat) - 1))
            p95_lat = sorted_lat[idx]

        return EvalSummary(
            total_samples=total,
            mean_faithfulness=round(mean_f, 4),
            mean_relevance=round(mean_r, 4),
            pass_rate=round(pass_rate, 4),
            pass_threshold=self.pass_threshold,
            p95_latency_ms=round(p95_lat, 2),
            results=results,
        )
