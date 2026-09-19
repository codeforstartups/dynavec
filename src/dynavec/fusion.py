"""Learned fusion-weight fitting for Reciprocal Rank Fusion (RRF).

``RRFWeightFitter`` learns per-retriever RRF weights by maximising mean nDCG
over a small set of labeled queries.  Three search strategies are available:

* **grid** – exhaustive simplex grid (exact for 2 retrievers, coarse for 3+).
* **random** – Dirichlet-sampled random search on the weight simplex.
* **bayesian** – Nelder-Mead on a softmax-parameterised simplex via
  ``scipy.optimize.minimize`` (falls back to *random* when scipy is absent).

Fitted weights are returned as a ``FitResult`` dataclass which can be
serialised to / loaded from JSON with ``save()`` / ``load()``.
"""

from __future__ import annotations

import itertools
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .eval import LabeledQuery, compute_ndcg_at_k
from .models import SearchResult
from .retrieval import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of an ``RRFWeightFitter.fit()`` call.

    Attributes
    ----------
    weights:
        Per-retriever weights (same order as *result_lists*).
    score:
        Best mean nDCG@k achieved by these weights.
    method:
        Search strategy used (``"grid"``, ``"random"``, or ``"bayesian"``).
    n_evaluations:
        Total number of weight candidates evaluated.
    """

    weights: list[float]
    score: float
    method: str
    n_evaluations: int

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON encoding."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist this result to a JSON file at *path*."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> FitResult:
        """Load a ``FitResult`` from a JSON file written by ``save()``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=data["weights"],
            score=float(data["score"]),
            method=data["method"],
            n_evaluations=int(data["n_evaluations"]),
        )


# ---------------------------------------------------------------------------
# RRFWeightFitter
# ---------------------------------------------------------------------------

class RRFWeightFitter:
    """Learn per-retriever RRF weights from labeled evaluation queries.

    Parameters
    ----------
    labeled_queries:
        Ground-truth queries with relevance labels.
    result_lists:
        Shape ``[n_retrievers][n_queries][hits]``.  Each retriever provides a
        ranked list of ``SearchResult`` objects for every labeled query.
    k:
        RRF smoothing constant (default ``60``).
    eval_k:
        Cutoff for nDCG evaluation (default ``10``).
    """

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

    # -- scoring -------------------------------------------------------------

    def score(self, weights: list[float]) -> float:
        """Mean nDCG@k across all labeled queries for *weights*."""
        ndcg_scores = []

        for query_idx, labeled_query in enumerate(self.labeled_queries):
            # result_lists[retriever_idx][query_idx] → list[SearchResult]
            per_retriever = [
                self.result_lists[r][query_idx]
                for r in range(len(self.result_lists))
            ]

            fused = reciprocal_rank_fusion(per_retriever, k=self.k, weights=weights)
            ranked_ids = [r.id for r in fused]

            relevant = labeled_query.relevance_grades or labeled_query.relevant_ids
            ndcg_scores.append(compute_ndcg_at_k(ranked_ids, relevant, k=self.eval_k))

        return float(np.mean(ndcg_scores))

    # -- fitting strategies --------------------------------------------------

    def _fit_grid(self, n_points: int) -> FitResult:
        """Exhaustive grid on the weight simplex.

        For 2 retrievers this is the classic ``w, 1-w`` sweep.  For *n*
        retrievers the simplex is discretised into ``n_points`` steps per
        dimension using a stars-and-bars partition.
        """
        n = len(self.result_lists)
        best_score = -1.0
        best_w: list[float] = [1.0 / n] * n
        evals = 0

        if n == 2:
            # Fast path: 1-D sweep (backward compatible)
            for i in range(n_points + 1):
                w1 = i / n_points
                w = [w1, 1.0 - w1]
                s = self.score(w)
                evals += 1
                if s > best_score:
                    best_score, best_w = s, w
        else:
            # N-D simplex grid via stars-and-bars
            steps = min(n_points, 20)  # cap steps per dimension for tractability
            for combo in itertools.combinations(range(steps + n - 1), n - 1):
                # Convert combination to weight partition
                parts = []
                prev = -1
                for c in combo:
                    parts.append(c - prev - 1)
                    prev = c
                parts.append(steps + n - 2 - prev)
                w = [p / steps for p in parts]
                s = self.score(w)
                evals += 1
                if s > best_score:
                    best_score, best_w = s, w

        return FitResult(weights=best_w, score=best_score, method="grid", n_evaluations=evals)

    def _fit_random(self, n_points: int) -> FitResult:
        """Dirichlet-sampled random search on the weight simplex."""
        n = len(self.result_lists)
        rng = np.random.default_rng(42)
        best_score = -1.0
        best_w: list[float] = [1.0 / n] * n

        for _ in range(n_points):
            raw = rng.exponential(scale=1.0, size=n)
            w = (raw / raw.sum()).tolist()
            s = self.score(w)
            if s > best_score:
                best_score, best_w = s, w

        return FitResult(weights=best_w, score=best_score, method="random", n_evaluations=n_points)

    def _fit_bayesian(self, n_points: int) -> FitResult:
        """Nelder-Mead on a softmax-parameterised simplex.

        Uses ``scipy.optimize.minimize`` when available; otherwise falls back
        to ``_fit_random`` with a warning.
        """
        try:
            from scipy.optimize import minimize  # type: ignore[import-untyped]
        except ImportError:
            warnings.warn(
                "scipy is not installed — falling back to method='random'. "
                "Install scipy for bayesian fitting: pip install scipy",
                stacklevel=2,
            )
            result = self._fit_random(n_points)
            return FitResult(
                weights=result.weights,
                score=result.score,
                method="bayesian",  # record intent
                n_evaluations=result.n_evaluations,
            )

        n = len(self.result_lists)
        eval_count = 0

        def _softmax(x: np.ndarray) -> list[float]:
            e = np.exp(x - x.max())
            return [float(value) for value in (e / e.sum()).tolist()]

        def _neg_ndcg(x: np.ndarray) -> float:
            nonlocal eval_count
            eval_count += 1
            return -self.score(_softmax(x))

        x0 = np.zeros(n)
        result = minimize(
            _neg_ndcg,
            x0,
            method="Nelder-Mead",
            options={"maxfev": n_points, "xatol": 1e-4, "fatol": 1e-6},
        )

        best_w = _softmax(result.x)
        best_score = -float(result.fun)

        return FitResult(weights=best_w, score=best_score, method="bayesian", n_evaluations=eval_count)

    # -- public API ----------------------------------------------------------

    def fit(self, method: str = "grid", n_points: int = 200) -> FitResult:
        """Find the best RRF weights for the labeled queries.

        Parameters
        ----------
        method:
            ``"grid"`` (default), ``"random"``, or ``"bayesian"``.
        n_points:
            Budget: number of grid divisions, random samples, or max
            function evaluations (depending on *method*).

        Returns
        -------
        FitResult
            A dataclass with ``.weights``, ``.score``, ``.method``, and
            ``.n_evaluations``.
        """
        if method == "grid":
            return self._fit_grid(n_points)
        if method == "random":
            return self._fit_random(n_points)
        if method == "bayesian":
            return self._fit_bayesian(n_points)
        raise ValueError(f"Unknown method {method!r}, expected 'grid', 'random', or 'bayesian'")
