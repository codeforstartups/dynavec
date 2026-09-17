
from __future__ import annotations

import numpy as np

from .eval import LabeledQuery, compute_ndcg_at_k
from .models import SearchResult
from .retrieval import reciprocal_rank_fusion


class RRFWeightFitter:
    def __init__(
        self,
        labeled_queries: list[LabeledQuery],
        result_lists: list[list[list[SearchResult]]],
        k: int = 60,
        eval_k: int = 10,
    ) -> None:
        if not labeled_queries:
            raise ValueError("labeled_queries must be non-empty")
        if not result_lists:
            raise ValueError("result_lists must be non-empty")
        self.labeled_queries = labeled_queries
        self.result_lists = result_lists
        self.k = k
        self.eval_k = eval_k


    def score(self, weights: list[float]) -> float:
        ndcg_scores = []

        for query_idx, labeled_query in enumerate(self.labeled_queries):
            # Get the result list for this query from each retriever
            # result_lists[retriever_idx][query_idx] → list[SearchResult]
            per_retriever = [
                self.result_lists[r][query_idx]
                for r in range(len(self.result_lists))
            ]

            # Fuse using the candidate weights
            fused = reciprocal_rank_fusion(per_retriever, k=self.k, weights=weights)

            # Extract the ranked doc IDs
            ranked_ids = [r.id for r in fused]

            # Score against ground truth
            relevant = labeled_query.relevance_grades or labeled_query.relevant_ids
            ndcg_scores.append(compute_ndcg_at_k(ranked_ids, relevant, k=self.eval_k))

        return float(np.mean(ndcg_scores))

    def _fit_grid(self, n_points: int) -> list[float]:
        best_score, best_w = -1.0, [1.0, 1.0]
        for i in range(n_points + 1):
            w1 = i / n_points
            w2 = 1.0 - w1
            s = self.score([w1, w2])
            if s > best_score:
                best_score, best_w = s, [w1, w2]
        return best_w

    def _fit_random(self, n_points: int) -> list[float]:
        n = len(self.result_lists)
        rng = np.random.default_rng(42)
        best_score, best_w = -1.0, [1.0 / n] * n
        for _ in range(n_points):
            raw = rng.exponential(scale=1.0, size=n)
            w = (raw / raw.sum()).tolist()
            s = self.score(w)
            if s > best_score:
                best_score, best_w = s, w
        return best_w

    def fit(self, method: str = "grid", n_points: int = 200) -> list[float]:
        if method == "grid":
            return self._fit_grid(n_points)
        if method == "random":
            return self._fit_random(n_points)
        raise ValueError(f"Unknown method {method!r}, expected 'grid' or 'random'")
