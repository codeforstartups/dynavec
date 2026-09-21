"""Evaluation framework for RAG faithfulness, answer relevance, and retrieval quality (Recall/MRR/nDCG)."""

from __future__ import annotations

from .base import (
    AnswerRelevanceResult,
    BaseJudge,
    ClaimVerification,
    FaithfulnessResult,
    RAGEvalResult,
    extract_json,
)
from .chart import plot_eval_summary, plot_retrieval_metrics
from .judges import (
    BedrockJudge,
    CustomJudge,
    GeminiJudge,
    MockJudge,
    OpenAIJudge,
)
from .metrics import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_rag,
)
from .retrieval import (
    LabeledQuery,
    RetrievalEvalRunner,
    RetrievalEvalSummary,
    RetrievalMetricResult,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from .runner import (
    EvalRunner,
    EvalSummary,
)
from .trend import EvalRun, EvalRunStore

__all__ = [
    "EvalRun",
    "EvalRunStore",
    # LLM Judge & Data Models
    "BaseJudge",
    "OpenAIJudge",
    "BedrockJudge",
    "GeminiJudge",
    "CustomJudge",
    "MockJudge",
    "ClaimVerification",
    "FaithfulnessResult",
    "AnswerRelevanceResult",
    "RAGEvalResult",
    "extract_json",
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "evaluate_rag",
    "EvalRunner",
    "EvalSummary",
    "plot_eval_summary",
    # Information Retrieval (IR) Evaluation
    "LabeledQuery",
    "RetrievalMetricResult",
    "RetrievalEvalSummary",
    "RetrievalEvalRunner",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "plot_retrieval_metrics",
]
