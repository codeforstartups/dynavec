"""Base abstractions and data models for LLM-as-a-Judge evaluation."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM output.

    Handles:
    - Raw JSON string: '{"score": 1.0}'
    - Markdown fenced code blocks: '```json\n{"score": 1.0}\n```'
    - Text with embedded JSON: 'Evaluation result: {"score": 1.0} Explanation: ...'
    """
    clean = text.strip()
    if not clean:
        raise ValueError("Empty response from LLM judge.")

    # 1. Try direct json parsing
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. Try markdown fenced code block (```json ... ``` or ``` ... ```)
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Try finding outermost matching braces { ... }
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = clean[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON object from LLM judge response:\n{clean}")


@dataclass
class ClaimVerification:
    """Verification result for a single atomic claim."""

    claim: str
    supported: bool
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Result of a faithfulness / groundedness evaluation."""

    score: float
    claims: list[ClaimVerification] = field(default_factory=list)
    reasoning: str = ""
    supported_count: int = 0
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerRelevanceResult:
    """Result of an answer relevance evaluation."""

    score: float
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGEvalResult:
    """Combined RAG evaluation result for a query, context, and answer."""

    query: str
    context: list[str]
    answer: str
    faithfulness: FaithfulnessResult | None = None
    answer_relevance: AnswerRelevanceResult | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseJudge(ABC):
    """Abstract base class for LLM judges.

    Subclasses must implement :meth:`judge` to generate raw text responses from
    the target LLM model.
    """

    @abstractmethod
    def judge(self, prompt: str) -> str:
        """Send a prompt to the LLM judge and return its raw response string."""

    def judge_structured(self, prompt: str) -> dict[str, Any]:
        """Send a prompt and extract a structured JSON dictionary from the response."""
        raw_text = self.judge(prompt)
        return extract_json(raw_text)
