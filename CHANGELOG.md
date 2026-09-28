# Changelog

All notable changes to dynavec are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Optimistic concurrency on `update()`** (#24) — each update stores a `version` on the
  DynamoDB item and writes with a `ConditionExpression`, so a concurrent change raises
  `ConflictError` (nothing written) instead of being silently overwritten. Pass
  `expected_version=` to guard across your own read/update cycle; `UpsertResult.version`
  returns the new version. Plain `upsert()` stays last-writer-wins and resets the version.

### Changed
- `update()` now writes DynamoDB before S3 Vectors (previously in parallel) so a conflict
  leaves both stores untouched, and reads the document with a strongly consistent `GetItem`.

## [0.6.0] - 2026-09-25

A large release: new retrieval strategies, quantization methods, graph and cache
capabilities, more integrations, and quality-of-life tooling.

### Added
- **Query-expansion retrievers** (#215) — `MultiQueryRetriever` (concurrent reformulation
  search + RRF) and `HyDERetriever` (hypothetical-document embeddings; average / fuse strategies).
- **Learned RRF fusion weights** (#204, #223) — `RRFWeightFitter` fits per-retriever RRF
  weights by maximizing nDCG (grid / random / optional Bayesian), with `FitResult` save/load.
- **Optimized Product Quantization** (#198) — `OPQRotation` + `OptimizedProductQuantizer`
  (rotate-then-PQ with Procrustes refinement); **ScalarQuantizer** INT8 (#201).
- **Cross-encoder reranking** (#208).
- **Hybrid graph + ANN search** (#206) — `hybrid_graph_search` fuses ANN and graph-scoped
  results with weighted RRF; **graph shortest path** (#237); **graph node/edge deletion** (#209).
- **`search().explain()`** debug output — per-stage timing and candidate counts (#228).
- **Query-cache invalidation on write** (#236) — `invalidate(namespace)` across backends,
  `cache_invalidate_on_write` config; **`warm_cache()`** to pre-populate the cache (#194);
  **optional embedding cache** (`CachedEmbedder`, #213).
- **Client `describe()`** returning `IndexInfo` diagnostics (#218); **`dynavec --version`** (#238).
- **CLI namespace export/import** to/from JSONL (#217).
- **Ingestion**: `CsvSource` + row-wise spreadsheets (#205); **`S3Source`** bucket ingestion (#61).
- **Hot-tier LRU/FIFO eviction** policy (#216).
- **DynamoDB**: gzip-compress large text before storing (#231); item-size validation (#211).
- **Integrations**: OpenAI Assistants file-search tool (#73), LlamaIndex metadata-filter
  translation (#219), Strands retriever example, multi-tenant RAG example/docs.
- **Embeddings**: OpenAI retry with `Retry-After` handling (#242); dimension > 4096 warning (#241).
- **Eval**: retrieval-quality run trend tracking (#212).
- **Site**: landing-page revamp + interactive cost calculator (#104).

### Changed
- **`XlsxSource` now yields one record per row** (header → `"column: value"`), not per
  worksheet (#205). Behavior change vs 0.5.0.
- `max_workers` is validated eagerly in config (#250).

## [0.5.0] - 2026-09-16

### Added
- **Office document ingestion** (#188) — `DocxSource`, `PptxSource`, and `XlsxSource`
  for Word, PowerPoint, and Excel files, each lazy-importing its parser.
- **Hugging Face Inference embedder** (#191) — `HFInferenceEmbedder` backed by the
  HF Serverless Inference API.
- **DSPy retrieval integration** (#195) — `DynavecRM(dspy.Retrieve)` so a dynavec
  client can back a DSPy pipeline (closes #68).
- **Structured logging** (#200) — opt-in `structured_logging=True` emits JSON store
  events with secret redaction; `log_level` config.
- **ProductQuantizer persistence** (#199) — `save()` / `load()` via `np.savez` with
  `allow_pickle=False` (safe serialization).
- **Dashboard dark mode** (#187) — theme toggle + parity with the landing page.

### Changed / Performance
- **Vectorized MMR** (#197) — reranking over large candidate sets is now
  O(k·N) instead of O(k·N·k).

### Tests
- Property-based tests for the metric layer via Hypothesis (#192), plus rescore
  metric-override coverage (#196).

## [0.4.0] - 2026-09-12


### Added
- **In-memory hot tier** (#186) — opt-in `hot_tier=True` keeps the hot working
  set in RAM via the built-in `SPFreshHotIndex`. After `db.warm(namespace)`, a
  namespace is served **entirely from memory** — no S3 Vectors query and no
  DynamoDB hydration — for in-memory-engine latency without a paid cluster.
  Write-through on upsert/update/delete keeps it current; filters, rescore, and
  MMR run on the hot path; non-authoritative namespaces fall back to S3 safely.
  New: `warm()`, `hot_stats()`, and `hot_tier*` config.
- **Retrieval quality runner** (#185) — recall@k, MRR, and nDCG@k evaluation.
- **Async LangChain retrieval** (#183) — `asimilarity_search` /
  `asimilarity_search_with_score` / `amax_marginal_relevance_search`, so the
  retriever `ainvoke()` path runs on an owned async surface (closes #70).
- **Graph export** (#182) — `graph_export()` to Mermaid and Graphviz DOT.
- **Ollama local embedder** (#180) and **URL ingestion source** (#181).
- **Markdown directory ingestion** (#174) and **DynamoDB cache TTL jitter** (#145).

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
