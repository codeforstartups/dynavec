"""Batch dataset evaluation runner for RAG quality benchmarking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Union

from .base import BaseJudge, RAGEvalResult
from .metrics import evaluate_rag

EvalTriplet = tuple[str, Union[str, Sequence[str]], str]


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
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0

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
        reference_answer: str | None = None,
        run_context_precision: bool = True,
        run_context_recall: bool = True,
    ) -> RAGEvalResult:
        """Evaluate a single query-context-answer triplet."""
        return evaluate_rag(
            query=query,
            context=context,
            answer=answer,
            judge=self.judge,
            run_faithfulness=run_faithfulness,
            run_answer_relevance=run_answer_relevance,
            reference_answer=reference_answer,
            run_context_precision=run_context_precision,
            run_context_recall=run_context_recall,
        )

    def run(
        self,
        dataset: Sequence[dict[str, Any] | EvalTriplet],
        run_faithfulness: bool = True,
        run_answer_relevance: bool = True,
        run_context_precision: bool = True,
        run_context_recall: bool = True,
    ) -> EvalSummary:
        """Evaluate an entire dataset in batch and compute aggregate quality metrics.

        Each item in `dataset` can be:
        - A dict with ``query``, ``context``, ``answer`` and optional ``reference_answer``
        - A legacy 3-tuple ``(query, context, answer)``

        Context precision and context recall run only when a reference answer is present.
        """
        results: list[RAGEvalResult] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        context_precision_scores: list[float] = []
        context_recall_scores: list[float] = []
        latencies: list[float] = []
        passed_count = 0

        for item in dataset:
            reference_answer: str | None = None

            if isinstance(item, dict):
                q = str(item.get("query", ""))
                c = item.get("context", [])
                a = str(item.get("answer", ""))

                raw_reference = item.get("reference_answer")
                if raw_reference is not None:
                    reference_answer = str(raw_reference)

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
                reference_answer=reference_answer,
                run_context_precision=run_context_precision,
                run_context_recall=run_context_recall,
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

            if res.context_precision is not None:
                context_precision_scores.append(res.context_precision.score)
                if res.context_precision.score < self.pass_threshold:
                    is_pass = False

            if res.context_recall is not None:
                context_recall_scores.append(res.context_recall.score)
                if res.context_recall.score < self.pass_threshold:
                    is_pass = False

            if is_pass and (
                res.faithfulness is not None
                or res.answer_relevance is not None
                or res.context_precision is not None
                or res.context_recall is not None
            ):
                passed_count += 1

        total = len(results)
        mean_f = (
            float(sum(faithfulness_scores) / len(faithfulness_scores))
            if faithfulness_scores
            else 0.0
        )
        mean_r = float(sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 0.0
        mean_cp = (
            float(sum(context_precision_scores) / len(context_precision_scores))
            if context_precision_scores
            else 0.0
        )
        mean_cr = (
            float(sum(context_recall_scores) / len(context_recall_scores))
            if context_recall_scores
            else 0.0
        )
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
            mean_context_precision=round(mean_cp, 4),
            mean_context_recall=round(mean_cr, 4),
        )
