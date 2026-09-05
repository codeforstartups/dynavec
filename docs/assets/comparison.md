# dynavec benchmark comparison

_Workload: 1,000,000 × 768d, 1,000,000 queries/mo._

| Metric | dynavec | Pinecone | OpenSearch | Qdrant | Weaviate | Milvus/Zilliz |
|---|---|---|---|---|---|---|
| Recall@10 | 0.90 | 0.95 | 0.97 | **0.98** | 0.97 | 0.98 |
| Latency p50 (ms) | 45 | 30 | 15 | 8 | 10 | **7** |
| Latency p95 (ms) | 120 | 70 | 40 | 20 | 25 | **18** |
| Cost ($/mo) | **$3** | $9 | $701 | $160 | $175 | $150 |
| Serverless (scale-to-zero) | Yes | Yes | No (OCU floor) | No (nodes) | No (nodes) | No (CU) |
| Data in your AWS account | Yes | No | Yes | Self-host only | Self-host only | Self-host only |
