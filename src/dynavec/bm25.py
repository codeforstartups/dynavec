"""Lightweight Okapi BM25 index and tokenizer for lexical retrieval.

Implements pure-Python Okapi BM25 with zero third-party dependencies.
Computes Robertson-Spärck Jones / Lucene IDF and BM25 term-frequency saturation
with length normalization, supporting exact keyword lookups, part numbers, SKUs,
and acronyms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .models import Document, Metadata, SearchResult

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_.:][a-zA-Z0-9]+)*")
_SUBTOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)


def default_tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    stopwords: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Tokenize text into search terms, preserving compound identifiers.

    Extracts compound identifiers (e.g. 'SKU-892-XZ', 'XPS-13-9310',
    'ConnectionResetError') as both full compound tokens and individual
    sub-tokens for maximal recall on both exact matches and partial phrases.
    """
    if not text:
        return []

    lowered = text.lower()
    stop_set = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    tokens: list[str] = []
    # 1) Extract compound patterns (e.g. sku-892-xz, 10.0.0.1)
    compounds = _TOKEN_RE.findall(lowered)
    for comp in compounds:
        if remove_stopwords and comp in stop_set:
            continue
        tokens.append(comp)
        # 2) If compound has delimiters, also emit individual sub-tokens
        if any(c in comp for c in "-_.:"):
            subtokens = _SUBTOKEN_RE.findall(comp)
            for sub in subtokens:
                if remove_stopwords and sub in stop_set:
                    continue
                tokens.append(sub)

    return tokens


def _matches_filter(doc_meta: Metadata, filter_spec: Metadata) -> bool:
    """Check if document metadata matches key-value filter conditions."""
    for key, expected in filter_spec.items():
        actual = doc_meta.get(key)
        if isinstance(expected, dict):
            # Operator support: e.g. {"$in": [...]}, {"$eq": val}
            for op, val in expected.items():
                if op == "$eq" and actual != val:
                    return False
                if op == "$ne" and actual == val:
                    return False
                if op == "$in" and (not isinstance(val, (list, tuple, set)) or actual not in val):
                    return False
                if op == "$nin" and isinstance(val, (list, tuple, set)) and actual in val:
                    return False
                if op == "$gt" and (actual is None or actual <= val):
                    return False
                if op == "$gte" and (actual is None or actual < val):
                    return False
                if op == "$lt" and (actual is None or actual >= val):
                    return False
                if op == "$lte" and (actual is None or actual > val):
                    return False
        elif actual != expected:
            return False
    return True


class BM25Index:
    """In-memory Okapi BM25 inverted index with zero third-party dependencies."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
        remove_stopwords: bool = True,
        stopwords: frozenset[str] | set[str] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or default_tokenize
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords

        # Stored documents: id -> {text, metadata, len, tf}
        self._docs: dict[str, dict[str, Any]] = {}
        # Term frequencies across corpus: term -> document count
        self._doc_freqs: Counter[str] = Counter()
        # Inverted index: term -> set(doc_id)
        self._inverted_index: dict[str, set[str]] = {}

        self._total_docs: int = 0
        self._total_length: int = 0
        self._avg_doc_len: float = 0.0
        self._idf_cache: dict[str, float] = {}

    def __len__(self) -> int:
        return self._total_docs

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is default_tokenize:
            return default_tokenize(
                text,
                remove_stopwords=self.remove_stopwords,
                stopwords=self.stopwords,
            )
        return self.tokenizer(text)

    def add_document(
        self,
        doc_id: str,
        text: str | None,
        metadata: Metadata | None = None,
    ) -> None:
        """Add or overwrite a document in the index."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")

        raw_text = text or ""
        tokens = self._tokenize(raw_text)
        tf = Counter(tokens)
        doc_len = len(tokens)

        # If overwriting existing document, remove it first
        if doc_id in self._docs:
            self.remove_document(doc_id)

        meta = dict(metadata or {})
        self._docs[doc_id] = {
            "text": raw_text,
            "metadata": meta,
            "len": doc_len,
            "tf": tf,
        }

        for term in tf:
            self._doc_freqs[term] += 1
            if term not in self._inverted_index:
                self._inverted_index[term] = set()
            self._inverted_index[term].add(doc_id)

        self._total_docs += 1
        self._total_length += doc_len
        self._avg_doc_len = self._total_length / self._total_docs if self._total_docs > 0 else 0.0
        self._idf_cache.clear()

    def add_documents(
        self,
        documents: Sequence[Document | dict[str, Any] | tuple[str, str | None]],
    ) -> None:
        """Batch-insert multiple documents into the index."""
        for d in documents:
            if isinstance(d, Document):
                self.add_document(d.id, d.text, d.metadata)
            elif isinstance(d, dict):
                self.add_document(
                    str(d["id"]),
                    d.get("text"),
                    d.get("metadata"),
                )
            elif isinstance(d, tuple):
                self.add_document(d[0], d[1])
            else:
                raise TypeError(f"Unsupported document type: {type(d)}")

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index. Returns True if removed, False if absent."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False

        doc_len = doc["len"]
        tf: Counter[str] = doc["tf"]

        for term in tf:
            self._doc_freqs[term] -= 1
            if self._doc_freqs[term] <= 0:
                del self._doc_freqs[term]
            if term in self._inverted_index:
                self._inverted_index[term].discard(doc_id)
                if not self._inverted_index[term]:
                    del self._inverted_index[term]

        self._total_docs -= 1
        self._total_length -= doc_len
        self._avg_doc_len = self._total_length / self._total_docs if self._total_docs > 0 else 0.0
        self._idf_cache.clear()
        return True

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._docs.clear()
        self._doc_freqs.clear()
        self._inverted_index.clear()
        self._total_docs = 0
        self._total_length = 0
        self._avg_doc_len = 0.0
        self._idf_cache.clear()

    def idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones / Lucene smoothed inverse document frequency."""
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached

        df = self._doc_freqs.get(term, 0)
        # Smoothing prevents negative weights on common terms
        score = math.log(1.0 + (self._total_docs - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = score
        return score

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        """Compute the BM25 score of a single document for tokenized query terms."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return 0.0

        tf = doc["tf"]
        doc_len = doc["len"]
        avg_len = self._avg_doc_len or 1.0

        score = 0.0
        for term in query_tokens:
            count = tf.get(term, 0)
            if count == 0:
                continue
            term_idf = self.idf(term)
            # Okapi BM25 TF saturation formula
            denom = count + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
            score += term_idf * (count * (self.k1 + 1.0)) / denom

        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: Metadata | None = None,
    ) -> list[SearchResult]:
        """Search the BM25 index and return ranked SearchResult items."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query or not query.strip() or self._total_docs == 0:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Find candidate documents containing at least one term
        candidate_ids: set[str] = set()
        for term in tokens:
            doc_ids = self._inverted_index.get(term)
            if doc_ids:
                candidate_ids.update(doc_ids)

        if not candidate_ids:
            return []

        scored: list[tuple[float, str]] = []
        for doc_id in candidate_ids:
            doc_entry = self._docs[doc_id]
            if filter and not _matches_filter(doc_entry["metadata"], filter):
                continue
            bm25_score = self.score(tokens, doc_id)
            if bm25_score > 0.0:
                scored.append((bm25_score, doc_id))

        # Sort descending by score; tie-break deterministically by doc_id
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[SearchResult] = []
        for score, doc_id in scored[:top_k]:
            doc_entry = self._docs[doc_id]
            results.append(
                SearchResult(
                    id=doc_id,
                    score=round(score, 6),
                    text=doc_entry["text"],
                    metadata=doc_entry["metadata"],
                )
            )

        return results
