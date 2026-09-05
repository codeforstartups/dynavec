"""Cost model: estimate \\$/month for dynavec vs managed/self-hosted competitors.

⚠️  Pricing changes constantly and varies by region/tier/commitment. The numbers
below are **approximate public list prices** meant for order-of-magnitude
comparison, not quotes. Update ``PRICING`` before citing anything externally.

Run:  python -m benchmarks.cost_model --vectors 5_000_000 --dim 768 --qpm 1_000_000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------- assumptions
# All prices are USD. Marked APPROX — verify against current vendor pricing.
GB = 1024**3


@dataclass
class Workload:
    vectors: int          # number of stored vectors
    dim: int              # embedding dimension
    queries_per_month: int
    writes_per_month: int = 0
    metadata_bytes: int = 500     # avg full metadata + text per doc (DynamoDB)
    filterable_bytes: int = 120   # avg filterable metadata per vector (S3 Vectors)

    @property
    def vector_bytes(self) -> int:
        return self.dim * 4  # float32

    @property
    def raw_vector_gb(self) -> float:
        return self.vectors * self.vector_bytes / GB

    @property
    def doc_store_gb(self) -> float:
        # DynamoDB holds text + metadata ONLY — the vector lives in S3 Vectors,
        # so it is never duplicated here.
        return self.vectors * self.metadata_bytes / GB


def dynavec_cost(w: Workload) -> dict[str, float]:
    """S3 Vectors (storage + query + put) + DynamoDB (storage + on-demand R/W).

    APPROX prices (us-east-1):
      S3 Vectors: ~$0.06 /GB-mo storage; query & ingest priced per request +
                  data scanned. We approximate query at ~$0.0025 / 1k queries
                  and ingest at ~$0.20 / GB uploaded (verify!).
      DynamoDB on-demand: $0.25 /GB-mo; $1.25 /1M writes; $0.25 /1M reads (eventually
                  consistent BatchGetItem counts as 0.5 RRU per 4KB item).
    """
    s3_storage = (w.raw_vector_gb + w.vectors * w.filterable_bytes / GB) * 0.06
    s3_query = (w.queries_per_month / 1000) * 0.0025
    s3_ingest = (w.writes_per_month * w.vector_bytes / GB) * 0.20

    ddb_storage = w.doc_store_gb * 0.25
    ddb_writes = (w.writes_per_month / 1_000_000) * 1.25
    ddb_reads = (w.queries_per_month / 1_000_000) * 0.25  # 1 BatchGet hydration/query

    total = s3_storage + s3_query + s3_ingest + ddb_storage + ddb_writes + ddb_reads
    return {
        "s3vectors_storage": s3_storage,
        "s3vectors_query": s3_query,
        "s3vectors_ingest": s3_ingest,
        "dynamodb_storage": ddb_storage,
        "dynamodb_writes": ddb_writes,
        "dynamodb_reads": ddb_reads,
        "TOTAL": total,
    }


def pinecone_serverless_cost(w: Workload) -> float:
    """APPROX Pinecone serverless: storage ~$0.33/GB-mo, reads ~$8.25/1M RU,
    writes ~$2/1M WU (1 RU ≈ small query). Very rough."""
    storage = w.raw_vector_gb * 0.33
    reads = (w.queries_per_month / 1_000_000) * 8.25
    writes = (w.writes_per_month / 1_000_000) * 2.0
    return storage + reads + writes


def opensearch_serverless_cost(w: Workload) -> float:
    """APPROX OpenSearch Serverless: ~$0.24/OCU-hr, min 4 OCU (2 index + 2 search)
    even when idle, and OCUs must scale with data kept hot for vector search.

    Model: OCUs = max(4, ceil(hot_gb / 6)), hot_gb = raw * 0.5 (rest tiers to
    disk/quantized). ~6 GB working memory per OCU. Plus cheap S3 storage.
    """
    import math

    hot_gb = w.raw_vector_gb * 0.5
    ocus = max(4, math.ceil(hot_gb / 6.0))
    compute = ocus * 730 * 0.24
    storage = w.raw_vector_gb * 0.024  # managed S3-backed storage
    return compute + storage


def _managed_node_cost(w: Workload, gb_per_node: float, node_monthly: float) -> float:
    """Generic self-hosted/managed cluster: size by RAM to hold vectors + overhead."""
    import math

    needed_gb = w.raw_vector_gb * 1.5  # HNSW graph + overhead
    nodes = max(1, math.ceil(needed_gb / gb_per_node))
    return nodes * node_monthly


def qdrant_cost(w: Workload) -> float:
    # APPROX Qdrant Cloud: ~$0.014/GB-RAM-hr; model as 16GB nodes @ ~$160/mo.
    return _managed_node_cost(w, gb_per_node=16, node_monthly=160)


def weaviate_cost(w: Workload) -> float:
    # APPROX Weaviate Cloud standard: model as 16GB nodes @ ~$175/mo.
    return _managed_node_cost(w, gb_per_node=16, node_monthly=175)


def milvus_zilliz_cost(w: Workload) -> float:
    # APPROX Zilliz Cloud dedicated: model as 16GB CU @ ~$150/mo.
    return _managed_node_cost(w, gb_per_node=16, node_monthly=150)


def compare(w: Workload) -> dict[str, float]:
    dv = dynavec_cost(w)
    return {
        "dynavec": dv["TOTAL"],
        "pinecone_serverless": pinecone_serverless_cost(w),
        "opensearch_serverless": opensearch_serverless_cost(w),
        "qdrant_cloud": qdrant_cost(w),
        "weaviate_cloud": weaviate_cost(w),
        "milvus_zilliz": milvus_zilliz_cost(w),
    }


def _fmt(d: dict[str, float]) -> str:
    rows = sorted(d.items(), key=lambda kv: kv[1])
    width = max(len(k) for k in d)
    lines = [f"{k:<{width}}  ${v:>12,.2f} / mo" for k, v in rows]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="dynavec cost comparison (APPROX list prices)")
    p.add_argument("--vectors", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--qpm", type=int, default=1_000_000, help="queries per month")
    p.add_argument("--wpm", type=int, default=100_000, help="writes per month")
    args = p.parse_args()

    w = Workload(vectors=args.vectors, dim=args.dim,
                 queries_per_month=args.qpm, writes_per_month=args.wpm)

    print(f"\nWorkload: {w.vectors:,} vectors × {w.dim}d "
          f"({w.raw_vector_gb:.1f} GB raw), {w.queries_per_month:,} queries/mo, "
          f"{w.writes_per_month:,} writes/mo\n")
    print("dynavec breakdown:")
    for k, v in dynavec_cost(w).items():
        print(f"  {k:<22} ${v:>10,.2f}")
    print("\nEstimated monthly cost (APPROX list prices — verify before quoting):")
    print(_fmt(compare(w)))
    print("\n⚠️  Order-of-magnitude only. See PRICING notes in cost_model.py.\n")


if __name__ == "__main__":
    main()
