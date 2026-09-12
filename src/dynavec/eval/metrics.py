"""Core RAG evaluation metrics: Faithfulness (hallucination detection) and Answer Relevance."""

from __future__ import annotations

import time
from collections.abc import Sequence

from .base import (
    AnswerRelevanceResult,
    BaseJudge,
    ClaimVerification,
    FaithfulnessResult,
    RAGEvalResult,
)

_FAITHFULNESS_PROMPT_TEMPLATE = """You are an expert evaluator assessing the FAITHFULNESS (groundedness) of a generated answer with respect to retrieved context.

Task:
1. Break down the generated answer into discrete, atomic factual statements/claims.
2. For each claim, determine whether it is directly ENTAILED and SUPPORTED by the provided context.
   - "supported": true if the context directly confirms or logically entails the claim.
   - "supported": false if the context contradicts, fails to mention, or only partially supports the claim.
3. Provide a brief reasoning for each claim.

Retrieved Context:
{context_text}

User Query:
{query}

Generated Answer:
{answer}

Respond ONLY with a JSON object in this exact schema:
{{
  "claims": [
    {{
      "claim": "<statement extracted from answer>",
      "supported": true,
      "reasoning": "<why this claim is or is not supported by the context>"
    }}
  ],
  "reasoning": "<overall summary of faithfulness>"
}}
"""

_RELEVANCE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the ANSWER RELEVANCE of a generated answer with respect to a user query.

Task:
1. Determine how directly, completely, and concisely the generated answer addresses the user's question or intent.
2. Penalize answers that contain irrelevant tangents, off-topic fluff, or fail to address the core question.
3. Assign a relevance score between 0.0 (completely irrelevant / non-responsive) and 1.0 (perfectly relevant and directly addresses query intent).

User Query:
{query}

Generated Answer:
{answer}

Respond ONLY with a JSON object in this exact schema:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation for the assigned score>"
}}
"""


def evaluate_faithfulness(
    query: str,
    context: str | Sequence[str],
    answer: str,
    judge: BaseJudge,
) -> FaithfulnessResult:
    """Evaluate whether all factual claims in the answer are grounded in the retrieved context.

    Parameters
    ----------
    query:
        The original user question or prompt.
    context:
        Retrieved context chunks (string or list of strings).
    answer:
        The generated answer text to evaluate.
    judge:
        The pluggable LLM judge instance.

    Returns
    -------
    FaithfulnessResult:
        Contains overall score (0.0 to 1.0), list of verified claims, and reasoning.
    """
    if not answer or not answer.strip():
        return FaithfulnessResult(
            score=0.0,
            claims=[],
            reasoning="Empty answer provided.",
            supported_count=0,
            total_count=0,
        )

    if isinstance(context, str):
        context_text = context.strip()
    else:
        context_text = "\n\n---\n\n".join(c.strip() for c in context if c.strip())

    if not context_text:
        # If there is an answer but no context, claims cannot be grounded in context
        return FaithfulnessResult(
            score=0.0,
            claims=[
                ClaimVerification(
                    claim=answer.strip(),
                    supported=False,
                    reasoning="No context provided to ground the answer.",
                )
            ],
            reasoning="Empty context provided; answer cannot be verified.",
            supported_count=0,
            total_count=1,
        )

    prompt = _FAITHFULNESS_PROMPT_TEMPLATE.format(
        context_text=context_text,
        query=query.strip() or "N/A",
        answer=answer.strip(),
    )

    data = judge.judge_structured(prompt)
    raw_claims = data.get("claims", [])
    reasoning = data.get("reasoning", "")

    verifications: list[ClaimVerification] = []
    supported_count = 0

    if isinstance(raw_claims, list) and raw_claims:
        for item in raw_claims:
            if isinstance(item, dict):
                claim_text = str(item.get("claim", "")).strip()
                is_supported = bool(item.get("supported", False))
                c_reason = str(item.get("reasoning", "")).strip()
                if claim_text:
                    if is_supported:
                        supported_count += 1
                    verifications.append(
                        ClaimVerification(
                            claim=claim_text,
                            supported=is_supported,
                            reasoning=c_reason,
                        )
                    )

    total_count = len(verifications)
    if total_count > 0:
        score = supported_count / total_count
    else:
        # Fallback to direct score field if claims list was empty
        score = float(data.get("score", 1.0))
        score = max(0.0, min(1.0, score))

    return FaithfulnessResult(
        score=round(score, 4),
        claims=verifications,
        reasoning=reasoning,
        supported_count=supported_count,
        total_count=total_count,
    )


def evaluate_answer_relevance(
    query: str,
    answer: str,
    judge: BaseJudge,
) -> AnswerRelevanceResult:
    """Evaluate how directly and concisely the answer addresses the user query.

    Parameters
    ----------
    query:
        The original user question or prompt.
    answer:
        The generated answer text.
    judge:
        The pluggable LLM judge instance.

    Returns
    -------
    AnswerRelevanceResult:
        Contains relevance score (0.0 to 1.0) and reasoning.
    """
    if not query or not query.strip():
        return AnswerRelevanceResult(
            score=0.0,
            reasoning="Empty query provided.",
        )
    if not answer or not answer.strip():
        return AnswerRelevanceResult(
            score=0.0,
            reasoning="Empty answer provided.",
        )

    prompt = _RELEVANCE_PROMPT_TEMPLATE.format(
        query=query.strip(),
        answer=answer.strip(),
    )

    data = judge.judge_structured(prompt)
    raw_score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, raw_score))
    reasoning = str(data.get("reasoning", ""))

    return AnswerRelevanceResult(
        score=round(score, 4),
        reasoning=reasoning,
    )


def evaluate_rag(
    query: str,
    context: str | Sequence[str],
    answer: str,
    judge: BaseJudge,
    run_faithfulness: bool = True,
    run_answer_relevance: bool = True,
) -> RAGEvalResult:
    """Perform combined RAG evaluation (Faithfulness + Answer Relevance).

    Parameters
    ----------
    query:
        The user query text.
    context:
        Retrieved context chunks (string or sequence of strings).
    answer:
        The generated answer text.
    judge:
        The pluggable LLM judge instance.
    run_faithfulness:
        Whether to evaluate faithfulness (default True).
    run_answer_relevance:
        Whether to evaluate answer relevance (default True).

    Returns
    -------
    RAGEvalResult:
        Contains individual metric results, query, context list, answer, and latency.
    """
    ctx_list = [context] if isinstance(context, str) else list(context)
    t0 = time.perf_counter()

    faithfulness_res = (
        evaluate_faithfulness(query=query, context=context, answer=answer, judge=judge)
        if run_faithfulness
        else None
    )

    relevance_res = (
        evaluate_answer_relevance(query=query, answer=answer, judge=judge)
        if run_answer_relevance
        else None
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return RAGEvalResult(
        query=query,
        context=ctx_list,
        answer=answer,
        faithfulness=faithfulness_res,
        answer_relevance=relevance_res,
        latency_ms=round(elapsed_ms, 2),
    )
