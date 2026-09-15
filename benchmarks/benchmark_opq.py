
"""Benchmark PQ vs OPQ + PQ."""

from __future__ import annotations

import time

import numpy as np

from dynavec.quantization import (
    OptimizedProductQuantizer,
    ProductQuantizer,
)


def exact_search(
    database: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return exact top-k nearest-neighbour indices."""
    results = []

    for query in queries:
        distances = ((database - query) ** 2).sum(axis=1)
        results.append(np.argsort(distances)[:k])

    return np.asarray(results)


def quantized_search(
    quantizer,
    codes: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """Search encoded vectors using ADC."""
    results = []

    for query in queries:
        distances = quantizer.asymmetric_distances(
            query,
            codes,
        )

        results.append(np.argsort(distances)[:k])

    return np.asarray(results)


def recall_at_k(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    k: int,
) -> float:
    """Calculate mean Recall@K."""
    recalls = []

    for truth, prediction in zip(
        ground_truth,
        predictions,
    ):
        truth_set = set(truth[:k])
        prediction_set = set(prediction[:k])

        recalls.append(
            len(truth_set & prediction_set) / k
        )

    return float(np.mean(recalls))


def benchmark_quantizer(
    name: str,
    quantizer,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Benchmark a quantizer."""
    print(f"\n--- {name} ---")

    start = time.perf_counter()

    quantizer.fit(database)

    fit_time = time.perf_counter() - start

    start = time.perf_counter()

    codes = quantizer.encode(database)

    encode_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = quantized_search(
        quantizer,
        codes,
        queries,
        k=10,
    )

    search_time = time.perf_counter() - start

    recall_1 = recall_at_k(
        ground_truth,
        predictions,
        k=1,
    )

    recall_10 = recall_at_k(
        ground_truth,
        predictions,
        k=10,
    )

    reconstruction_error = quantizer.reconstruction_error(
        database
    )

    print(f"Fit time          : {fit_time:.4f}s")
    print(f"Encode time       : {encode_time:.4f}s")
    print(f"Search time       : {search_time:.4f}s")
    print(f"Recall@1          : {recall_1:.4f}")
    print(f"Recall@10         : {recall_10:.4f}")
    print(f"Reconstruction    : {reconstruction_error:.6f}")
    print(f"Bytes/vector      : {quantizer.code_size_bytes}")

    return {
        "recall_1": recall_1,
        "recall_10": recall_10,
        "reconstruction_error": reconstruction_error,
        "fit_time": fit_time,
        "encode_time": encode_time,
        "search_time": search_time,
    }


def main() -> None:
    rng = np.random.default_rng(42)

    num_database = 2000
    num_queries = 100
    dimension = 64

    m = 8
    nbits = 8

    database = rng.normal(
        size=(num_database, dimension)
    ).astype(np.float32)

    queries = rng.normal(
        size=(num_queries, dimension)
    ).astype(np.float32)

    print("=== PQ vs OPQ + PQ Benchmark ===")
    print(f"Database : {num_database} vectors")
    print(f"Queries  : {num_queries}")
    print(f"Dimension: {dimension}")
    print(f"m        : {m}")
    print(f"nbits    : {nbits}")

    # ---------------------------------------------------------
    # Exact ground truth
    # ---------------------------------------------------------
    start = time.perf_counter()

    ground_truth = exact_search(
        database,
        queries,
        k=10,
    )

    exact_time = time.perf_counter() - start

    print(f"\nExact search time: {exact_time:.4f}s")

    # ---------------------------------------------------------
    # Baseline PQ
    # ---------------------------------------------------------
    pq = ProductQuantizer(
        m=m,
        nbits=nbits,
        iters=25,
        seed=42,
    )

    pq_results = benchmark_quantizer(
        "PQ",
        pq,
        database,
        queries,
        ground_truth,
    )

    # ---------------------------------------------------------
    # OPQ + PQ
    # ---------------------------------------------------------
    opq_pq = OptimizedProductQuantizer(
        m=m,
        nbits=nbits,
        iters=25,
        opq_iters=5,
        seed=42,
    )

    opq_results = benchmark_quantizer(
        "OPQ + PQ",
        opq_pq,
        database,
        queries,
        ground_truth,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------
    print("\nOPQ training errors:")
    for i, error in enumerate(opq_pq.training_errors, start=1):
        print(f"Iteration {i}: {error:.6f}")

    print("\n=== Comparison ===")

    recall_improvement = (
        opq_results["recall_10"]
        - pq_results["recall_10"]
    )

    error_improvement = (
        pq_results["reconstruction_error"]
        - opq_results["reconstruction_error"]
    )

    print(
        f"Recall@10 improvement : "
        f"{recall_improvement:+.4f}"
    )

    print(
        f"Reconstruction improvement: "
        f"{error_improvement:+.6f}"
    )


if __name__ == "__main__":
    main()
