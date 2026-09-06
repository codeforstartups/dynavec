# Choosing an embedding dimension

A larger embedding dimension gives the ANN index more signal to separate neighbors — usually higher recall — but it also makes every vector bigger, which raises storage cost and (slightly) query latency. The right choice is the **smallest dimension that meets your recall target**, not the largest one available.

## The tradeoff in a nutshell

| Dimension | Example model | Raw size/vector | Recall trend | Cost trend | Best for |
|-----------|---------------|-----------------|--------------|------------|----------|
| **384** | `all-MiniLM-L6-v2`, `cohere-embed-v3` | 1.5 KB | Good on clean, short text | Lowest storage + bandwidth | High-volume, latency-sensitive, cost-first workloads |
| **768** | `all-mpnet-base-v2`, `bge-base` | 3 KB | Solid general-purpose | Moderate | General RAG, balanced quality/cost |
| **1024** | `bge-large`, `embed-english-v3` | 4 KB | Strong on nuanced semantic similarity | Moderate-high | Complex domains (legal, medical, code) |
| **1536** | `text-embedding-3-small`, `ada-002` | 6 KB | High on most benchmarks | Higher | Default for OpenAI users; strong recall |
| **3072** | `text-embedding-3-large` | 12 KB | Highest among OpenAI models | Highest storage + latency | Maximum recall; small-to-medium corpora |
| **4096** | S3 Vectors maximum | 16 KB | Upper bound | Max | Rarely needed; only if your model requires it |

> **Rule of thumb:** start at 768 or 1536, measure recall@k on your own queries, and only go higher if the gain justifies the storage cost. For most RAG applications, 768-dim models (e.g., BGE-base) deliver 95%+ of the recall at half the storage of 1536-dim models.

## S3 Vectors constraint

Amazon S3 Vectors caps the **maximum dimension per vector at 4096**. Any model whose output exceeds this (e.g., some 7680-dim embeddings) cannot be indexed directly — you must reduce dimensionality first (PCA, random projection, or choose a smaller model). All mainstream embedding models (384 – 3072) fit within this limit.

## Shortening: when a smaller model is "good enough"

OpenAI's `text-embedding-3-small` (1536-dim) is the default choice for many teams, but if your corpus is large and your recall target is modest (e.g., 0.85+ recall@10 on FAQ-style chunks), a 768-dim open model often matches it at **half the storage cost** and with lower latency. The repo's `benchmarks/` directory lets you measure this on your own data:

```bash
python -m benchmarks.run_benchmark --backend dynavec \
    --bucket my-vectors --index bench --table dynavec_bench \
    --n 100000 --dim 768   # swap to 1536 to compare
```

## Cost impact by dimension

From the repo's cost model (1M vectors, 1M queries/month, public list prices):

| Dimension | dynavec $/mo | Pinecone $/mo | Raw storage |
|-----------|--------------|---------------|-------------|
| 384 | **$3** | $9 | 1.4 TB @ 1B vectors |
| 768 | **$3** | $9 | 2.9 TB @ 1B vectors |
| 1024 | **$3** | $10 | 3.8 TB @ 1B vectors |
| 1536 | **$3** | $10 | 5.7 TB @ 1B vectors |
| 3072 | **$4** | $12 | 11.4 TB @ 1B vectors |

dynavec's storage is priced like S3, so the dimension multiplier matters less than on RAM-based systems — but it still shows up in per-query bandwidth and in the non-S3 line items (DynamoDB, request costs). See [scaling.md](assets/scaling.md) for the full 100K → 1B sweep across all five dimensions.

## Picking in practice

1. **Set a recall target** (e.g., 0.90 recall@10) on a labeled eval set.
2. **Start with 768-dim** (BGE-base or similar). Measure.
3. **If recall is below target**, step up to 1024 or 1536. Re-measure.
4. **Stop when the target is met** — going higher costs more for diminishing returns.
5. **If cost matters more than peak recall**, stay at the smallest dimension that gets you within ~5% of your target.
