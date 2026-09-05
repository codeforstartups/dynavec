"""Benchmark runner: recall@k + latency (p50/p95/p99), local or against AWS.

Backends
--------
--backend local   In-memory brute-force pipeline. No AWS, no cost. Validates
                  pipeline correctness and gives latency of the non-AWS layers.
                  (Recall is ~1.0 by construction — the ANN engine is exact here.)
--backend dynavec Real DynamoDB + S3 Vectors in your account. Measures true
                  end-to-end latency and S3 Vectors recall. Requires AWS creds,
                  --bucket, --index, --table.

Examples
--------
    python -m benchmarks.run_benchmark --backend local --n 20000 --dim 384
    python -m benchmarks.run_benchmark --backend dynavec \
        --bucket my-vectors --index bench --table dynavec_bench \
        --n 100000 --dim 1024 --region us-east-1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .datasets import make_synthetic, recall_at_k


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


# --------------------------------------------------------------- local backend
class LocalBruteForce:
    """Exact in-memory NN — a control/baseline, no AWS."""

    def __init__(self):
        self._vecs = None
        self._ids = None

    def build(self, vectors, ids):
        self._vecs = vectors
        self._ids = ids

    def query(self, q, k):
        sims = self._vecs @ q
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx.tolist()


# ------------------------------------------------------------- dynavec backend
class DynavecBackend:
    def __init__(self, args):
        from dynavec import Document, Dynavec, DynavecConfig

        self._Document = Document
        cfg = DynavecConfig(
            vector_bucket=args.bucket,
            index=args.index,
            table=args.table,
            dimension=args.dim,
            region=args.region,
            auto_provision=True,
        )
        self._db = Dynavec(cfg)  # BYO vectors: no embedder needed
        self._ns = "bench"
        self._id_to_index = {}

    def build(self, vectors, ids):
        self._id_to_index = {vid: i for i, vid in enumerate(ids)}
        docs = [
            self._Document(id=vid, vector=vectors[i].tolist())
            for i, vid in enumerate(ids)
        ]
        # upsert in batches so we don't build one giant list in memory
        B = 500
        for s in range(0, len(docs), B):
            self._db.upsert(docs[s : s + B], namespace=self._ns)

    def query(self, q, k):
        hits = self._db.search(vector=q.tolist(), top_k=k, namespace=self._ns)
        return [self._id_to_index[h.id] for h in hits if h.id in self._id_to_index]


def run(args) -> None:
    print(f"Generating synthetic dataset: {args.n:,} × {args.dim}d, "
          f"{args.queries} queries, k={args.k}")
    ds = make_synthetic(n=args.n, dim=args.dim, n_queries=args.queries, k=args.k)

    if args.backend == "local":
        backend = LocalBruteForce()
    else:
        backend = DynavecBackend(args)

    print(f"Building index ({args.backend}) ...")
    t0 = time.perf_counter()
    backend.build(ds.vectors, ds.ids)
    build_s = time.perf_counter() - t0
    print(f"  build: {build_s:.2f}s")

    print("Querying ...")
    latencies = []
    retrieved = []
    for q in ds.queries:
        t = time.perf_counter()
        idx = backend.query(q, args.k)
        latencies.append((time.perf_counter() - t) * 1000.0)
        retrieved.append(idx)

    recall = recall_at_k(retrieved, ds.ground_truth, args.k)
    pct = _percentiles(latencies)

    print("\n================ RESULTS ================")
    print(f"backend         : {args.backend}")
    print(f"dataset         : {args.n:,} vectors × {args.dim}d")
    print(f"recall@{args.k:<8}: {recall:.4f}")
    print(f"latency (ms)    : p50={pct['p50']:.2f}  p95={pct['p95']:.2f}  "
          f"p99={pct['p99']:.2f}  mean={pct['mean']:.2f}")
    print(f"build time      : {build_s:.2f}s")
    if args.backend == "local":
        print("\nNote: local backend is exact NN (recall≈1.0). Use --backend "
              "dynavec for real S3 Vectors recall & end-to-end latency.")
    print("=========================================\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["local", "dynavec"], default="local")
    p.add_argument("--n", type=int, default=10_000, help="number of vectors")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    # dynavec backend only
    p.add_argument("--bucket")
    p.add_argument("--index", default="bench")
    p.add_argument("--table", default="dynavec_bench")
    p.add_argument("--region")
    args = p.parse_args()

    if args.backend == "dynavec" and not args.bucket:
        p.error("--backend dynavec requires --bucket (and AWS credentials)")
    run(args)


if __name__ == "__main__":
    main()
