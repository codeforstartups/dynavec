# dynavec benchmarks

Three things to measure when replacing a vector DB: **recall**, **latency**, and **cost**. This suite covers all three.

```bash
pip install "dynavec[benchmark]"
```

## 1. Recall + latency

```bash
# Local control run (no AWS): validates the pipeline, gives non-AWS-layer latency.
python -m benchmarks.run_benchmark --backend local --n 50000 --dim 384

# Real end-to-end against S3 Vectors + DynamoDB in your account:
python -m benchmarks.run_benchmark --backend dynavec \
    --bucket my-vectors --index bench --table dynavec_bench \
    --region us-east-1 --n 100000 --dim 1024 --queries 500 --k 10
```

Reports `recall@k` (vs exact brute-force ground truth) and latency `p50/p95/p99/mean`.

- **`--backend local`** is an exact in-memory NN baseline — recall is ~1.0 by construction. Use it to sanity-check the pipeline and to profile the non-AWS layers, **not** to judge S3 Vectors recall.
- **`--backend dynavec`** measures the real thing: true S3 Vectors recall and full round-trip latency including DynamoDB hydration. Needs AWS credentials and incurs (small) AWS charges.

The synthetic dataset is **clustered** (random cluster centers + noise), which is far more representative of real embeddings than uniform noise. Swap in your own vectors by editing `datasets.py`.

## 2. Cost comparison

```bash
python -m benchmarks.cost_model --vectors 5_000_000 --dim 768 --qpm 2_000_000 --wpm 200_000
```

Estimates \$/month for dynavec vs Pinecone (serverless), OpenSearch Serverless, Qdrant Cloud, Weaviate Cloud, and Milvus/Zilliz.

> ⚠️ **Pricing is approximate.** The constants in `cost_model.py` are public list prices for order-of-magnitude comparison, and they go stale. Update `PRICING` before quoting anything externally. The structural point holds regardless: dynavec has **no idle floor** (you pay storage + per-request), whereas cluster- and OCU-based systems bill for provisioned capacity around the clock.

## 3. Report (tables + charts)

```bash
python -m benchmarks.report --vectors 1_000_000 --dim 768 --qpm 1_000_000
```

Writes to `benchmarks/out/`:
- `comparison.md` — a comparison table (best value per row **bolded**)
- `quality_latency.png` — recall + p50/p95/p99 latency bars
- `cost_by_scale.png` — cost vs number of vectors on log-log axes

Cost is computed by the real `cost_model`. Recall/latency for competitors are
**representative constants** in `report.py` (`_PROFILE`) — replace them with your
measured numbers from `run_benchmark.py --backend dynavec` before publishing.

## Running a fair comparison

To benchmark competitors head-to-head, run each on the **same dataset and query set** (`datasets.make_synthetic(...)` with a fixed seed) and record recall@k + latency the same way. Keep embedding out of the timed loop (embed once, store vectors) so you measure the index, not the embedder.
