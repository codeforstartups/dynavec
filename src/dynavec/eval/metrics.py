"""Core RAG evaluation metrics: faithfulness, answer relevance, context precision, and context recall."""

from __future__ import annotations

import time
from collections.abc import Sequence

from .base import (
    AnswerRelevanceResult,
    BaseJudge,
    ClaimVerification,
    ContextPrecisionResult,
    ContextRecallResult,
    ContextRelevanceVerification,
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


_CONTEXT_PRECISION_PROMPT_TEMPLATE = """You are an expert evaluator assessing CONTEXT PRECISION for a RAG system.

Task:
Evaluate each retrieved context chunk independently and determine whether it is useful for answering the user query according to the reference answer.

A context is relevant when it contains information that helps support or produce the reference answer.
A context is irrelevant when it does not help answer the query or contains unrelated information.

Preserve the original context rank.

User Query:
{query}

Reference Answer:
{reference_answer}

Retrieved Contexts:
{contexts_text}

Respond ONLY with a JSON object in this exact schema:
{{
  "contexts": [
    {{
      "index": 1,
      "relevant": true,
      "reasoning": "<why this context is or is not useful>"
    }}
  ],
  "reasoning": "<overall summary of context precision>"
}}
"""


_CONTEXT_RECALL_PROMPT_TEMPLATE = """You are an expert evaluator assessing CONTEXT RECALL for a RAG system.

Task:
1. Break the reference answer into discrete, atomic factual claims.
2. For each claim, determine whether it is supported by the retrieved context.
3. A claim is supported when the retrieved context contains enough information to establish that claim.
4. Mark unsupported claims when the required information is missing from the retrieved context.

Retrieved Context:
{context_text}

User Query:
{query}

Reference Answer:
{reference_answer}

Respond ONLY with a JSON object in this exact schema:
{{
  "claims": [
    {{
      "claim": "<atomic factual claim from the reference answer>",
      "supported": true,
      "reasoning": "<why the retrieved context does or does not support this claim>"
    }}
  ],
  "reasoning": "<overall summary of context recall>"
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


def evaluate_context_precision(
    query: str,
    context: str | Sequence[str],
    reference_answer: str,
    judge: BaseJudge,
) -> ContextPrecisionResult:
    """Evaluate whether relevant retrieved contexts are ranked ahead of irrelevant ones."""

    if isinstance(context, str):
        contexts = [context.strip()] if context.strip() else []
    else:
        contexts = [chunk.strip() for chunk in context if chunk.strip()]

    if not contexts:
        return ContextPrecisionResult(
            score=0.0,
            contexts=[],
            relevant_count=0,
            total_count=0,
            reasoning="Empty context provided.",
        )

    if not reference_answer or not reference_answer.strip():
        return ContextPrecisionResult(
            score=0.0,
            contexts=[
                ContextRelevanceVerification(
                    context=chunk,
                    relevant=False,
                    reasoning="No reference answer provided.",
                )
                for chunk in contexts
            ],
            relevant_count=0,
            total_count=len(contexts),
            reasoning="Empty reference answer provided.",
        )

    contexts_text = "\n\n".join(
        f"[{index}] {chunk}" for index, chunk in enumerate(contexts, start=1)
    )

    prompt = _CONTEXT_PRECISION_PROMPT_TEMPLATE.format(
        query=query.strip() or "N/A",
        reference_answer=reference_answer.strip(),
        contexts_text=contexts_text,
    )

    data = judge.judge_structured(prompt)
    raw_contexts = data.get("contexts", [])
    reasoning = str(data.get("reasoning", ""))

    verdicts_by_index: dict[int, dict[str, object]] = {}

    if isinstance(raw_contexts, list):
        for position, item in enumerate(raw_contexts, start=1):
            if not isinstance(item, dict):
                continue

            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError):
                index = position

            if 1 <= index <= len(contexts):
                verdicts_by_index[index] = item

    verifications: list[ContextRelevanceVerification] = []
    relevant_count = 0
    precision_sum = 0.0

    for rank, chunk in enumerate(contexts, start=1):
        item = verdicts_by_index.get(rank, {})
        relevant = item.get("relevant", False) is True
        chunk_reasoning = str(item.get("reasoning", "")).strip()

        if relevant:
            relevant_count += 1
            precision_sum += relevant_count / rank

        verifications.append(
            ContextRelevanceVerification(
                context=chunk,
                relevant=relevant,
                reasoning=chunk_reasoning,
            )
        )

    score = precision_sum / relevant_count if relevant_count > 0 else 0.0

    return ContextPrecisionResult(
        score=round(score, 4),
        contexts=verifications,
        relevant_count=relevant_count,
        total_count=len(contexts),
        reasoning=reasoning,
    )


def evaluate_context_recall(
    query: str,
    context: str | Sequence[str],
    reference_answer: str,
    judge: BaseJudge,
) -> ContextRecallResult:
    """Evaluate how much of the reference answer is supported by retrieved context."""

    if not reference_answer or not reference_answer.strip():
        return ContextRecallResult(
            score=0.0,
            claims=[],
            supported_count=0,
            total_count=0,
            reasoning="Empty reference answer provided.",
        )

    if isinstance(context, str):
        context_text = context.strip()
    else:
        context_text = "\n\n---\n\n".join(chunk.strip() for chunk in context if chunk.strip())

    if not context_text:
        return ContextRecallResult(
            score=0.0,
            claims=[
                ClaimVerification(
                    claim=reference_answer.strip(),
                    supported=False,
                    reasoning="No context provided to support the reference answer.",
                )
            ],
            supported_count=0,
            total_count=1,
            reasoning="Empty context provided.",
        )

    prompt = _CONTEXT_RECALL_PROMPT_TEMPLATE.format(
        context_text=context_text,
        query=query.strip() or "N/A",
        reference_answer=reference_answer.strip(),
    )

    data = judge.judge_structured(prompt)
    raw_claims = data.get("claims", [])
    reasoning = str(data.get("reasoning", ""))

    verifications: list[ClaimVerification] = []
    supported_count = 0

    if isinstance(raw_claims, list):
        for item in raw_claims:
            if not isinstance(item, dict):
                continue

            claim = str(item.get("claim", "")).strip()
            supported = item.get("supported", False) is True
            claim_reasoning = str(item.get("reasoning", "")).strip()

            if not claim:
                continue

            if supported:
                supported_count += 1

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    supported=supported,
                    reasoning=claim_reasoning,
                )
            )

    total_count = len(verifications)
    score = supported_count / total_count if total_count > 0 else 0.0

    return ContextRecallResult(
        score=round(score, 4),
        claims=verifications,
        supported_count=supported_count,
        total_count=total_count,
        reasoning=reasoning,
    )


def evaluate_rag(
    query: str,
    context: str | Sequence[str],
    answer: str,
    judge: BaseJudge,
    run_faithfulness: bool = True,
    run_answer_relevance: bool = True,
    reference_answer: str | None = None,
    run_context_precision: bool = True,
    run_context_recall: bool = True,
) -> RAGEvalResult:
    """Perform combined RAG evaluation across generation and context quality metrics.

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
    reference_answer:
        Optional labeled/reference answer required for context metrics.
    run_context_precision:
        Whether to evaluate context precision when a reference answer is provided.
    run_context_recall:
        Whether to evaluate context recall when a reference answer is provided.

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

    has_reference = bool(reference_answer and reference_answer.strip())

    context_precision_res = (
        evaluate_context_precision(
            query=query,
            context=context,
            reference_answer=reference_answer or "",
            judge=judge,
        )
        if run_context_precision and has_reference
        else None
    )

    context_recall_res = (
        evaluate_context_recall(
            query=query,
            context=context,
            reference_answer=reference_answer or "",
            judge=judge,
        )
        if run_context_recall and has_reference
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
        reference_answer=reference_answer,
        context_precision=context_precision_res,
        context_recall=context_recall_res,
    )
