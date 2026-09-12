"""LLM-as-a-Judge evaluation framework for RAG faithfulness and answer relevance."""

from __future__ import annotations

from .base import (
    AnswerRelevanceResult,
    BaseJudge,
    ClaimVerification,
    FaithfulnessResult,
    RAGEvalResult,
    extract_json,
)
from .chart import plot_eval_summary
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
from .runner import (
    EvalRunner,
    EvalSummary,
)

__all__ = [
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
]
