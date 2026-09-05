"""Retrieval-side algorithms: score normalization, RRF fusion, and MMR rerank.

These are the parts of the pipeline dynavec owns end-to-end (S3 Vectors owns the
ANN itself). They're pure functions over results/vectors so they're trivially
testable without AWS.
"""

from __future__ import annotations

import numpy as np

from .config import DistanceMetric
from .models import SearchResult


def distance_to_score(distance: float, metric: DistanceMetric) -> float:
    """Normalize a raw distance so that **higher = more similar**.

    - cosine: S3 Vectors returns ``1 - cosine_similarity`` in ``[0, 2]`` →
      score = ``1 - distance`` (i.e. the cosine similarity itself).
    - euclidean: monotonic decreasing map into ``(0, 1]`` via ``1 / (1 + d)``.
    """
    if metric == "cosine":
        return 1.0 - distance
    return 1.0 / (1.0 + distance)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Fuse multiple ranked result lists with Reciprocal Rank Fusion.

    RRF is the standard way to combine a dense (vector) ranking with a sparse
    (keyword/BM25) ranking, or results from several indexes, without needing the
    scores to be on the same scale. ``score = sum(w / (k + rank))``.

    The fused ``SearchResult.score`` is the RRF score; original ``distance`` and
    fields are preserved from the first list a document appears in.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights length must match number of result lists")

    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for weight, results in zip(weights, result_lists):
        for rank, res in enumerate(results):
            scores[res.id] = scores.get(res.id, 0.0) + weight / (k + rank + 1)
            if res.id not in fused:
                fused[res.id] = res

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    out: list[SearchResult] = []
    for doc_id in ranked_ids:
        res = fused[doc_id]
        out.append(
            SearchResult(
                id=res.id,
                score=scores[doc_id],
                distance=res.distance,
                text=res.text,
                metadata=res.metadata,
                vector=res.vector,
            )
        )
    return out


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: list[SearchResult],
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[SearchResult]:
    """Re-rank candidates for relevance *and* diversity (MMR).

    Requires each candidate to carry its ``vector`` (dynavec fetches these via
    S3 Vectors ``get_vectors`` when MMR is requested). ``lambda_mult=1`` is pure
    relevance; ``0`` is pure diversity.
    """
    usable = [c for c in candidates if c.vector is not None]
    if not usable:
        return candidates[:top_k]

    q = np.asarray(query_vector, dtype=np.float32)
    mat = np.asarray([c.vector for c in usable], dtype=np.float32)

    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-12, None)

    qn = _norm(q.reshape(1, -1))[0]
    mn = _norm(mat)

    query_sim = mn @ qn  # cosine similarity to the query
    doc_sim = mn @ mn.T  # pairwise cosine similarity

    selected: list[int] = []
    remaining = set(range(len(usable)))
    top_k = min(top_k, len(usable))

    while len(selected) < top_k:
        best_idx = None
        best_score = -np.inf
        for i in remaining:
            diversity = max((doc_sim[i][j] for j in selected), default=0.0)
            score = lambda_mult * query_sim[i] - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [usable[i] for i in selected]
