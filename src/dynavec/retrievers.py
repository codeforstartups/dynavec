"""Query-expansion retrievers: Multi-Query and HyDE, fused with RRF.

Single-query vector search misses documents when the query is short, colloquial
or uses different vocabulary than the corpus. These two adapters widen the
candidate pool *before* fusion:

* :class:`MultiQueryRetriever` asks an LLM for a few reformulations of the query,
  searches with the original and every reformulation in parallel, and fuses the
  ranked lists with :func:`~dynavec.retrieval.reciprocal_rank_fusion`.
* :class:`HyDERetriever` (Hypothetical Document Embeddings) asks an LLM to write a
  short *answer passage*, embeds that passage as a **document**, and searches with
  the resulting vector -- optionally fused with the plain query search.

Both take plain callables, so dynavec stays free of LLM dependencies::

    from dynavec import MultiQueryRetriever

    retriever = MultiQueryRetriever(
        db.namespace("docs"),
        generate_queries=lambda q: my_llm_variations(q, n=3),
        top_k=4,
    )
    hits = retriever.search("how does serverless vector storage work?")

Notes
-----
* ``SearchResult.score`` on fused results is the **RRF score** (roughly 0.01-0.03),
  *not* a cosine similarity, and it is on that scale even when the LLM call fails
  and only the original query is searched. ``distance`` comes from the first list
  a document appears in; the original query's list is always first.
* Each sub-search is a normal :meth:`Dynavec.search` call, so each one pays for
  one S3 Vectors query plus one DynamoDB hydration of ``per_query_k`` documents.
  Cost per retriever call is therefore roughly ``(1 + n_queries)`` searches.
* Do not call ``retriever.search`` from a task already running on the client's
  internal executor: the fan-out submits into that same pool.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from .exceptions import ConfigurationError
from .models import Metadata, SearchResult
from .namespace import NamespaceView
from .retrieval import reciprocal_rank_fusion

if TYPE_CHECKING:  # pragma: no cover - typing only (client imports retrieval)
    from .client import Dynavec

logger = logging.getLogger(__name__)

OnGenerateError = Literal["fallback", "raise"]

__all__ = ["QueryExpansionRetriever", "MultiQueryRetriever", "HyDERetriever"]


def _clean_texts(original: str, candidates: object, limit: int) -> list[str]:
    """Normalize LLM output into a small list of distinct, non-blank strings.

    Accepts a list/tuple of strings (or a single string), strips whitespace,
    drops blanks and non-strings, removes case-insensitive duplicates (including
    duplicates of ``original``), and keeps at most ``limit`` entries in order.
    """
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        return []
    seen = {original.strip().casefold()}
    out: list[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        text = item.strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


class QueryExpansionRetriever:
    """Shared plumbing: source resolution, fan-out, RRF fusion, error policy."""

    def __init__(
        self,
        source: Dynavec | NamespaceView,
        *,
        namespace: str = "default",
        top_k: int = 4,
        per_query_k: int | None = None,
        rrf_k: int = 60,
        include_original: bool = True,
        original_weight: float = 1.0,
        on_generate_error: OnGenerateError = "fallback",
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if per_query_k is not None and per_query_k < 1:
            raise ValueError("per_query_k must be >= 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if original_weight <= 0:
            raise ValueError("original_weight must be > 0")
        if on_generate_error not in ("fallback", "raise"):
            raise ValueError("on_generate_error must be 'fallback' or 'raise'")

        if isinstance(source, NamespaceView):
            # Same package: unwrap the view so we can reach the client's thread
            # pool and embedder. The view's namespace always wins.
            self._db = source._db
            self._namespace = source.namespace
        else:
            self._db = source
            self._namespace = namespace

        self.top_k = top_k
        self.per_query_k = per_query_k
        self.rrf_k = rrf_k
        self.include_original = include_original
        self.original_weight = original_weight
        self.on_generate_error = on_generate_error

    # ------------------------------------------------------------- public API
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: Metadata | None = None,
        use_cache: bool | None = None,
    ) -> list[SearchResult]:
        """Expand ``query``, search, fuse with RRF, and return the top results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")
        depth = self.per_query_k if self.per_query_k is not None else 2 * k
        depth = max(depth, k)

        # Each entry is (weight, zero-arg callable returning one ranked list).
        plan = self._plan(query, depth, filter, use_cache)
        lists = self._fan_out([call for _, call in plan])
        weights = [w for w, _ in plan]
        return reciprocal_rank_fusion(lists, k=self.rrf_k, weights=weights)[:k]

    async def asearch(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: Metadata | None = None,
        use_cache: bool | None = None,
    ) -> list[SearchResult]:
        """Async wrapper (runs :meth:`search` in a worker thread)."""
        return await asyncio.to_thread(
            self.search, query, top_k=top_k, filter=filter, use_cache=use_cache
        )

    # ------------------------------------------------------------- internals
    def _plan(
        self,
        query: str,
        depth: int,
        filter: Metadata | None,
        use_cache: bool | None,
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raise NotImplementedError

    def _text_search(
        self, text: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> Callable[[], list[SearchResult]]:
        def call() -> list[SearchResult]:
            return self._db.search(
                text,
                top_k=depth,
                namespace=self._namespace,
                filter=filter,
                use_cache=use_cache,
            )

        return call

    def _vector_search(
        self,
        vector: Sequence[float],
        depth: int,
        filter: Metadata | None,
        use_cache: bool | None,
    ) -> Callable[[], list[SearchResult]]:
        def call() -> list[SearchResult]:
            return self._db.search(
                vector=list(vector),
                top_k=depth,
                namespace=self._namespace,
                filter=filter,
                use_cache=use_cache,
            )

        return call

    def _fan_out(self, calls: list[Callable[[], list[SearchResult]]]) -> list[list[SearchResult]]:
        """Run sub-searches concurrently; results keep submission order.

        Collecting futures in submission order (not completion order) is what
        makes fusion deterministic: RRF breaks score ties by first appearance.
        """
        if len(calls) == 1:
            return [calls[0]()]
        futures = [self._db._executor.submit(call) for call in calls]
        return [f.result() for f in futures]

    def _generate(self, fn: Callable[[str], object], query: str) -> object | None:
        """Call the user's LLM callable under the configured error policy."""
        try:
            return fn(query)
        except Exception as exc:  # noqa: BLE001 - user callable, any failure
            if self.on_generate_error == "raise":
                raise
            logger.warning(
                "%s: generation failed (%s: %s); falling back to the original query only",
                type(self).__name__,
                type(exc).__name__,
                exc,
            )
            return None


class MultiQueryRetriever(QueryExpansionRetriever):
    """Search with an LLM's reformulations of the query and fuse with RRF.

    Parameters
    ----------
    source:
        A :class:`~dynavec.client.Dynavec` client (use ``namespace=``) or a
        :class:`~dynavec.namespace.NamespaceView` (its namespace is used).
    generate_queries:
        ``Callable[[str], list[str]]`` returning alternative phrasings. Output
        is stripped, de-duplicated (case-insensitive, also against the original)
        and capped at ``n_queries``.
    n_queries:
        Maximum number of reformulations to use (extras are dropped).
    include_original:
        Also search with the original query (recommended: guards against poor
        reformulations). If generation fails, the original is always searched.
    original_weight:
        RRF weight of the original query's list; reformulations weigh ``1.0``.
    top_k / per_query_k / rrf_k:
        Final result count, candidates fetched per sub-search (default
        ``2 * top_k``; never less than ``top_k``), and the RRF constant.
    on_generate_error:
        ``"fallback"`` (default) logs a warning and searches the original query
        only; ``"raise"`` propagates the callable's exception. Errors from the
        stores themselves always propagate.
    """

    def __init__(
        self,
        source: Dynavec | NamespaceView,
        generate_queries: Callable[[str], Sequence[str]],
        *,
        n_queries: int = 3,
        **kwargs,
    ) -> None:
        if not callable(generate_queries):
            raise TypeError("generate_queries must be callable")
        if n_queries < 1:
            raise ValueError("n_queries must be >= 1")
        super().__init__(source, **kwargs)
        self._generate_queries = generate_queries
        self.n_queries = n_queries

    def _plan(self, query, depth, filter, use_cache):
        raw = self._generate(self._generate_queries, query)
        expansions = _clean_texts(query, raw, self.n_queries)

        plan: list[tuple[float, Callable[[], list[SearchResult]]]] = []
        # Original goes first so `distance` and tie-breaks favor it.
        if self.include_original or not expansions:
            plan.append((self.original_weight, self._text_search(query, depth, filter, use_cache)))
        for text in expansions:
            plan.append((1.0, self._text_search(text, depth, filter, use_cache)))
        return plan


class HyDERetriever(QueryExpansionRetriever):
    """Hypothetical Document Embeddings retrieval, optionally fused with the query.

    The hypothetical passage is embedded with ``embedder.embed_documents`` (the
    *document* side of asymmetric models) and searched by vector.

    Parameters
    ----------
    source:
        A ``Dynavec`` client or ``NamespaceView`` that has an embedder configured.
    generate_hypothetical:
        ``Callable[[str], str]`` returning a short passage that would answer the
        query. Blank output is treated as a generation failure.
    include_original:
        Fuse in the plain query search (default ``True``). Recommended: a
        hallucinated passage can point away from the right documents.
    original_weight, top_k, per_query_k, rrf_k, on_generate_error:
        As for :class:`MultiQueryRetriever`.
    """

    def __init__(
        self,
        source: Dynavec | NamespaceView,
        generate_hypothetical: Callable[[str], str],
        **kwargs,
    ) -> None:
        if not callable(generate_hypothetical):
            raise TypeError("generate_hypothetical must be callable")
        super().__init__(source, **kwargs)
        if self._db.embedder is None:
            raise ConfigurationError(
                "HyDERetriever needs an embedder to embed the hypothetical passage. "
                "Pass one to Dynavec(..., embedder=...)."
            )
        self._generate_hypothetical = generate_hypothetical

    def _plan(self, query, depth, filter, use_cache):
        raw = self._generate(self._generate_hypothetical, query)
        cleaned = _clean_texts("", raw, 1)
        hypothetical = cleaned[0] if cleaned else None

        plan: list[tuple[float, Callable[[], list[SearchResult]]]] = []
        if self.include_original or hypothetical is None:
            plan.append((self.original_weight, self._text_search(query, depth, filter, use_cache)))
        if hypothetical is not None:
            # Embedding happens here (calling thread) so embedder errors surface
            # directly instead of inside a pool future.
            vector = self._db.embedder.embed_documents([hypothetical])[0]
            plan.append((1.0, self._vector_search(vector, depth, filter, use_cache)))
        return plan
