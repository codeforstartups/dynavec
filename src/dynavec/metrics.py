"""Distance / similarity metrics for client-side rescoring & fusion.

The S3 Vectors *index* metric is cosine or euclidean (AWS-managed). On top of
the ANN candidate set, dynavec can rescore with any of these — or a weighted
**combination** — to squeeze quality for a given workload:

    cosine      angle only (scale-invariant); great for normalized embeddings
    dot         raw inner product; rewards magnitude (e.g. learned weights)
    euclidean   L2 straight-line distance
    manhattan   L1 / taxicab distance; robust to outlier dimensions

All scorers return **higher = more similar** so they compose cleanly.
"""

from __future__ import annotations

import numpy as np

Metric = str  # "cosine" | "dot" | "euclidean" | "manhattan"
_VALID = ("cosine", "dot", "euclidean", "manhattan")


def _as2d(mat: np.ndarray) -> np.ndarray:
    return mat if mat.ndim == 2 else mat.reshape(1, -1)


def score(query: np.ndarray, mat: np.ndarray, metric: Metric) -> np.ndarray:
    """Similarity of each row of ``mat`` to ``query`` (higher = closer)."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    m = _as2d(np.asarray(mat, dtype=np.float32))

    if metric == "dot":
        return m @ q
    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + 1e-12)
        mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        return mn @ qn
    if metric == "euclidean":
        d = np.linalg.norm(m - q, axis=1)
        return 1.0 / (1.0 + d)
    if metric == "manhattan":
        d = np.abs(m - q).sum(axis=1)
        return 1.0 / (1.0 + d)
    raise ValueError(f"unknown metric {metric!r}; expected one of {_VALID}")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a result set to the inclusive ``[0, 1]`` range."""
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        return x.copy()
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def composite_score(
    query: np.ndarray,
    mat: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted combination of several metrics.

    Each metric is min-max normalized across the candidate set before weighting,
    so metrics on different scales (e.g. dot vs cosine) combine fairly.

        composite_score(q, M, {"cosine": 0.7, "manhattan": 0.3})
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    total = 0.0
    acc = None
    for metric, w in weights.items():
        s = normalize_scores(score(query, mat, metric))
        acc = s * w if acc is None else acc + s * w
        total += w
    return acc / (total or 1.0)


def rescore(
    query: np.ndarray,
    candidate_vectors: np.ndarray,
    spec: Metric | dict[str, float],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Return an ordering (indices, best first) for the candidates under ``spec``.

    ``spec`` is a metric name or a ``{metric: weight}`` combination.
    Set ``normalize`` to min-max normalize the returned score array to ``[0, 1]``.
    """
    if isinstance(spec, str):
        scores = score(query, candidate_vectors, spec)
    else:
        scores = composite_score(query, candidate_vectors, spec)
    if normalize:
        scores = normalize_scores(scores)
    return np.argsort(-scores), scores
