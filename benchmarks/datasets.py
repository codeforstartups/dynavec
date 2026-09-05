"""Datasets + ground-truth for benchmarking recall and latency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    vectors: np.ndarray      # (N, dim) float32
    queries: np.ndarray      # (Q, dim) float32
    ids: list[str]
    ground_truth: np.ndarray  # (Q, k) int indices into vectors (exact NN)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


def make_synthetic(
    n: int = 10_000,
    dim: int = 384,
    n_queries: int = 200,
    k: int = 10,
    n_clusters: int = 50,
    seed: int = 0,
) -> Dataset:
    """Clustered synthetic vectors (more realistic than pure uniform noise).

    Vectors are drawn around random cluster centers; queries are perturbed
    cluster centers. Ground truth is exact cosine top-k by brute force.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)

    assign = rng.integers(0, n_clusters, size=n)
    vectors = centers[assign] + 0.35 * rng.normal(size=(n, dim)).astype(np.float32)
    vectors = _l2_normalize(vectors)

    q_assign = rng.integers(0, n_clusters, size=n_queries)
    queries = centers[q_assign] + 0.30 * rng.normal(size=(n_queries, dim)).astype(np.float32)
    queries = _l2_normalize(queries)

    ground_truth = _exact_topk(vectors, queries, k)
    ids = [f"vec-{i}" for i in range(n)]
    return Dataset(vectors=vectors, queries=queries, ids=ids, ground_truth=ground_truth)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(n, 1e-12, None)).astype(np.float32)


def _exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    # cosine similarity == dot product on L2-normalized vectors
    sims = queries @ vectors.T                       # (Q, N)
    return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]


def recall_at_k(retrieved_ids: list[list[int]], ground_truth: np.ndarray, k: int) -> float:
    """Mean recall@k across queries."""
    hits = 0
    total = 0
    for i, retrieved in enumerate(retrieved_ids):
        truth = set(ground_truth[i][:k].tolist())
        hits += len(truth.intersection(retrieved[:k]))
        total += min(k, len(truth))
    return hits / total if total else 0.0
