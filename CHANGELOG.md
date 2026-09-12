# Changelog

All notable changes to dynavec are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-10

### Added
- **Async embeddings** — `aembed_documents` / `aembed_query` on every backend,
  offloading sync clients to threads via `asyncio.to_thread` (#172).
- **New embedding backends** — Mistral (#152), Voyage AI (#149), and Bedrock
  Titan **multimodal image** embeddings (#173).
- **SPFresh hot-tier** — incremental hot-index rebalancing for freshly upserted
  vectors (#170).
- **PDF ingestion source** for the document pipeline (#169).
- **FastMCP server** exposing semantic and graph search as MCP tools (#165).
- **Observability** — native telemetry recorder + a stdlib dashboard on real
  query data, plus a full Next.js + TypeScript + Tailwind + Recharts dashboard
  under `dashboard/`.
- **`max_pool_connections`** config, threaded into every boto3 client for
  high-concurrency workloads (#175).
- `list_vectors` maintenance iterator for the S3 Vectors store (#166).
- Optional **score normalization** on search results (#147).
- Cache `hits` / `misses` counters and a `stats()` method on cache backends (#121).
- **AWS doctor** command for diagnosing credentials/permissions (#144).
- Ingestion **chunk deduplication** by content hash (#120).
- `py.typed` marker — dynavec now ships as a typed package (#154).
- Docs: FAQ, S3 Vectors regional availability matrix, embedding-dimension guide,
  IAM setup guide; pre-commit config + contributing guide (#160).

### Changed
- Semantic cache is now bounded by **bytes** rather than entry count (#146).

### Fixed
- Escape structured-storage key components to avoid namespace/id collisions (#140).
- Drain `QueryVectors` pages fully and add a `page_size` control (#151).
- Pin `crewai` away from the yanked 1.14.0 release (#161).
- Resolve optional dependencies correctly on Python 3.9 (#1).

## [0.2.0] - 2026-08

- Initial public release: hybrid DynamoDB + Amazon S3 Vectors store, pluggable
  embedders, namespace RAG, product quantization, RRF fusion, MMR rerank,
  GraphRAG layer, caching backends, framework adapters, and provisioning.
