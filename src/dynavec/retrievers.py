"""Query-expansion retrievers: Multi-Query and HyDE, fused with RRF.

Single-query vector search misses documents when the query is short, colloquial,
or uses different vocabulary than the corpus. These two adapters widen the
candidate pool *before* fusion:

* :class:`MultiQueryRetriever` asks an LLM for diverse reformulations of the query,
  searches with the original and every reformulation in parallel, and fuses the
  ranked lists with :func:`~dynavec.retrieval.reciprocal_rank_fusion`.
* :class:`HyDERetriever` (Hypothetical Document Embeddings) asks an LLM to write one
  or more *answer passages*, embeds them as **documents** (via ``embed_documents``),
  and searches with the resulting vector -- either averaging embeddings into a
  single centroid vector (the classic HyDE paper approach) or fusing multi-search
  results via RRF.

Both take plain callables (sync or async), so dynavec stays free of LLM dependencies::

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
* Sub-searches run concurrently over Dynavec's internal thread pool, while result
  lists strictly maintain submission order for deterministic tie-breaking.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .exceptions import ConfigurationError
from .models import Metadata, SearchResult
from .namespace import NamespaceView
from .retrieval import reciprocal_rank_fusion

if TYPE_CHECKING:
    from .client import Dynavec

logger = logging.getLogger(__name__)

OnGenerateError = Literal["fallback", "raise"]
HyDEStrategy = Literal["average", "fuse"]

__all__ = ["QueryExpansionRetriever", "MultiQueryRetriever", "HyDERetriever"]


def _clean_texts(original: str, candidates: object, limit: int) -> list[str]:
    """Normalize LLM output into a small list of distinct, non-blank strings.

    Accepts a string or sequence of strings, strips whitespace, drops non-strings
    and blanks, removes case-insensitive duplicates (including duplicates of
    ``original``), and keeps at most ``limit`` entries in order.
    """
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple, set, Sequence)):
        return []
    seen = {original.strip().casefold()} if original else set()
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
        include_original: bool | None = None,
        include_original_query: bool | None = None,
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

        inc_orig = True
        if include_original is not None:
            inc_orig = bool(include_original)
        elif include_original_query is not None:
            inc_orig = bool(include_original_query)

        if isinstance(source, NamespaceView):
            self._db = source._db
            self._namespace = source.namespace
        else:
            self._db = source
            self._namespace = namespace

        self.top_k = top_k
        self.per_query_k = per_query_k
        self.rrf_k = rrf_k
        self.include_original = inc_orig
        self.original_weight = original_weight
        self.on_generate_error = on_generate_error
        self._local_executor: ThreadPoolExecutor | None = None

    @property
    def _executor(self) -> ThreadPoolExecutor:
        if hasattr(self._db, "_executor") and self._db._executor is not None:
            return self._db._executor
        if self._local_executor is None:
            self._local_executor = ThreadPoolExecutor(max_workers=8)
        return self._local_executor

    # ------------------------------------------------------------- internals
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
        """Run sub-searches concurrently; results keep submission order."""
        if not calls:
            return []
        if len(calls) == 1:
            return [calls[0]()]
        futures = [self._executor.submit(call) for call in calls]
        return [f.result() for f in futures]

    def _invoke_generator(self, fn: Callable[[str], Any], query: str) -> Any:
        try:
            res = fn(query)
            if inspect.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        res = pool.submit(asyncio.run, res).result()
                else:
                    res = asyncio.run(res)
            return res
        except Exception as exc:
            if self.on_generate_error == "raise":
                raise
            logger.warning(
                "%s: generation failed (%s: %s); falling back to original query",
                type(self).__name__,
                type(exc).__name__,
                exc,
            )
            return None

    async def _async_invoke_generator(self, fn: Callable[[str], Any], query: str) -> Any:
        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(query)
            res = fn(query)
            if inspect.iscoroutine(res):
                return await res
            return res
        except Exception as exc:
            if self.on_generate_error == "raise":
                raise
            logger.warning(
                "%s: generation failed (%s: %s); falling back to original query",
                type(self).__name__,
                type(exc).__name__,
                exc,
            )
            return None

    def _plan(
        self,
        query: str,
        depth: int,
        filter: Metadata | None,
        use_cache: bool | None,
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raise NotImplementedError

    async def _async_plan(
        self,
        query: str,
        depth: int,
        filter: Metadata | None,
        use_cache: bool | None,
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        return await asyncio.to_thread(self._plan, query, depth, filter, use_cache)

    # ------------------------------------------------------------- public API
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: Metadata | None = None,
        use_cache: bool | None = None,
        weights: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """Expand ``query``, search, fuse with RRF, and return the top results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")
        depth = self.per_query_k if self.per_query_k is not None else 2 * k
        depth = max(depth, k)

        plan = self._plan(query, depth, filter, use_cache)
        lists = self._fan_out([call for _, call in plan])
        plan_weights = [w for w, _ in plan]
        fused_weights = list(weights) if weights is not None else plan_weights
        return reciprocal_rank_fusion(lists, k=self.rrf_k, weights=fused_weights)[:k]

    async def asearch(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: Metadata | None = None,
        use_cache: bool | None = None,
        weights: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """Async search: expands query (awaiting async LLMs) and executes searches."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")
        depth = self.per_query_k if self.per_query_k is not None else 2 * k
        depth = max(depth, k)

        plan = await self._async_plan(query, depth, filter, use_cache)
        lists = await asyncio.to_thread(self._fan_out, [call for _, call in plan])
        plan_weights = [w for w, _ in plan]
        fused_weights = list(weights) if weights is not None else plan_weights
        return reciprocal_rank_fusion(lists, k=self.rrf_k, weights=fused_weights)[:k]


class MultiQueryRetriever(QueryExpansionRetriever):
    """Search with an LLM's reformulations of the query and fuse with RRF.

    Parameters
    ----------
    source:
        A :class:`~dynavec.client.Dynavec` client (use ``namespace=``) or a
        :class:`~dynavec.namespace.NamespaceView` (its namespace is used).
    generate_queries:
        ``Callable[[str], Sequence[str]]`` returning alternative phrasings. Also
        accepts ``llm_generate_queries``. Output is stripped, de-duplicated
        (case-insensitive, also against the original), and capped at ``n_queries``.
    n_queries:
        Maximum number of reformulations to use (extras are dropped).
    include_original:
        Also search with the original query (recommended). Also accepts
        ``include_original_query``. If generation fails, original is always searched.
    original_weight:
        RRF weight of the original query's list; reformulations weigh ``1.0``.
    top_k / per_query_k / rrf_k:
        Final result count, candidates fetched per sub-search (default
        ``2 * top_k``; never less than ``top_k``), and the RRF constant.
    on_generate_error:
        ``"fallback"`` (default) logs a warning and searches the original query
        only; ``"raise"`` propagates the callable's exception.
    """

    def __init__(
        self,
        source: Dynavec | NamespaceView,
        generate_queries: Callable[[str], Sequence[str]] | None = None,
        *,
        llm_generate_queries: Callable[[str], Sequence[str]] | None = None,
        n_queries: int = 3,
        **kwargs: Any,
    ) -> None:
        gen = generate_queries if generate_queries is not None else llm_generate_queries
        if gen is None or not callable(gen):
            raise TypeError("generate_queries (or llm_generate_queries) must be a callable")
        if n_queries < 1:
            raise ValueError("n_queries must be >= 1")
        super().__init__(source, **kwargs)
        self._generate_queries = gen
        self.n_queries = n_queries

    def _build_plan(
        self, raw: Any, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        expansions = _clean_texts(query, raw, self.n_queries)
        plan: list[tuple[float, Callable[[], list[SearchResult]]]] = []
        if self.include_original or not expansions:
            plan.append((self.original_weight, self._text_search(query, depth, filter, use_cache)))
        for text in expansions:
            plan.append((1.0, self._text_search(text, depth, filter, use_cache)))
        return plan

    def _plan(
        self, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raw = self._invoke_generator(self._generate_queries, query)
        return self._build_plan(raw, query, depth, filter, use_cache)

    async def _async_plan(
        self, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raw = await self._async_invoke_generator(self._generate_queries, query)
        return self._build_plan(raw, query, depth, filter, use_cache)


class HyDERetriever(QueryExpansionRetriever):
    """Hypothetical Document Embeddings retrieval, optionally fused with the query.

    The hypothetical passage is embedded with ``embedder.embed_documents`` (the
    *document* side of asymmetric models) and searched by vector.

    Supports generating multiple hypothetical passages:
    - ``strategy="average"`` (default Centroid HyDE): embeds all passages, computes
      the normalized centroid vector, and executes a single vector search.
    - ``strategy="fuse"``: searches with each hypothetical vector independently and
      fuses their ranked lists via RRF.

    Parameters
    ----------
    source:
        A ``Dynavec`` client or ``NamespaceView`` that has an embedder configured.
    generate_hypothetical:
        ``Callable[[str], str | Sequence[str]]`` returning hypothetical passage(s).
        Also accepts ``llm_generate_hypothetical``.
    strategy:
        ``"average"`` (Centroid HyDE) or ``"fuse"`` (multi-search RRF).
    max_passages:
        Maximum number of hypothetical passages to embed if multiple are generated (default 5).
    include_original:
        Fuse in the plain query search (default ``True``). Also accepts
        ``include_original_query``.
    original_weight, top_k, per_query_k, rrf_k, on_generate_error:
        As for :class:`MultiQueryRetriever`.
    """

    def __init__(
        self,
        source: Dynavec | NamespaceView,
        generate_hypothetical: Callable[[str], str | Sequence[str]] | None = None,
        *,
        llm_generate_hypothetical: Callable[[str], str | Sequence[str]] | None = None,
        strategy: HyDEStrategy = "average",
        max_passages: int = 5,
        **kwargs: Any,
    ) -> None:
        gen = (
            generate_hypothetical
            if generate_hypothetical is not None
            else llm_generate_hypothetical
        )
        if gen is None or not callable(gen):
            raise TypeError(
                "generate_hypothetical (or llm_generate_hypothetical) must be a callable"
            )
        if strategy not in ("average", "fuse"):
            raise ValueError("strategy must be 'average' or 'fuse'")
        if max_passages < 1:
            raise ValueError("max_passages must be >= 1")
        super().__init__(source, **kwargs)
        if getattr(self._db, "embedder", None) is None:
            raise ConfigurationError(
                "HyDERetriever requires an embedder to embed hypothetical passages. "
                "Pass one to Dynavec(..., embedder=...)."
            )
        self._generate_hypothetical = gen
        self.strategy = strategy
        self.max_passages = max_passages

    def _build_plan(
        self, raw: Any, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        cleaned = _clean_texts("", raw, self.max_passages)
        plan: list[tuple[float, Callable[[], list[SearchResult]]]] = []

        if self.include_original or not cleaned:
            plan.append((self.original_weight, self._text_search(query, depth, filter, use_cache)))

        if cleaned:
            assert self._db.embedder is not None
            vectors = self._db.embedder.embed_documents(cleaned)
            if self.strategy == "average":
                if len(vectors) == 1:
                    centroid = vectors[0]
                else:
                    arr = np.mean(vectors, axis=0)
                    norm = float(np.linalg.norm(arr))
                    if norm > 1e-12:
                        arr = arr / norm
                    centroid = arr.tolist()
                plan.append((1.0, self._vector_search(centroid, depth, filter, use_cache)))
            else:  # "fuse"
                for vec in vectors:
                    plan.append((1.0, self._vector_search(vec, depth, filter, use_cache)))

        return plan

    def _plan(
        self, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raw = self._invoke_generator(self._generate_hypothetical, query)
        return self._build_plan(raw, query, depth, filter, use_cache)

    async def _async_plan(
        self, query: str, depth: int, filter: Metadata | None, use_cache: bool | None
    ) -> list[tuple[float, Callable[[], list[SearchResult]]]]:
        raw = await self._async_invoke_generator(self._generate_hypothetical, query)
        return self._build_plan(raw, query, depth, filter, use_cache)
