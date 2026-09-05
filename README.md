# dynavec

**Serverless, in-your-own-account hybrid vector database on AWS.**
`dynavec` fuses **Amazon DynamoDB** (single-digit-millisecond metadata + document store) with **Amazon S3 Vectors** (billion-scale, AWS-managed approximate-nearest-neighbor search) into one Python client — a drop-in alternative to Pinecone, Qdrant, Milvus, Weaviate, and OpenSearch that **runs entirely inside your AWS account** and **bills only when you use it**.

```bash
pip install dynavec                       # base: boto3 + numpy only
pip install "dynavec[openai]"             # + OpenAI embedder
pip install "dynavec[sentence-transformers]"  # + local/offline embedder
pip install "dynavec[all]"                # every embedder + LangChain
```

---

## Why dynavec

| Goal | How dynavec delivers it |
|------|-------------------------|
| **Cost-effective** | No always-on servers, no managed-service premium. You pay S3 Vectors storage/query + DynamoDB on-demand. Idle cost ≈ storage only. |
| **Lowest latency** | ANN keys come from S3 Vectors; the **actual documents are hydrated from DynamoDB via `BatchGetItem` in single-digit ms**. Warm S3 Vectors queries land ~100 ms. |
| **Scale** | S3 Vectors is designed to search across **billions of vectors** with 90%+ recall. |
| **Data compliance** | Every byte stays in **your** account, **your** region, **your** AZs. dynavec only ever calls AWS with your credentials. No third-party data plane. |
| **Secure / elastic** | Serverless primitives scale to zero and back automatically; IAM is the only access boundary. |

### What dynavec is *not* pretending to be

S3 Vectors **is** the ANN engine — AWS manages the index internally, so you don't (and can't) choose HNSW vs SPANN vs SPFresh there. dynavec's algorithmic value is the layers **around** it that you *do* control: the two-store hybrid design, metadata pre-filtering, **RRF hybrid fusion**, **MMR diversity reranking**, namespace/partition routing, and (on the roadmap) an optional in-process `hnswlib` **hot tier** for sub-10-ms hot-partition queries. See [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Architecture at a glance

```
             ┌──────────────────────── your AWS account ────────────────────────┐
  upsert ───▶│  Embedder (BYO key: OpenAI / Gemini / Cohere / Bedrock / local)   │
             │        │                                                          │
             │        ▼                                                          │
             │  ┌─────────────┐   vector + small filterable metadata            │
             │  │ S3 Vectors  │◀──────────────────────────────────┐            │
             │  │ (ANN index) │                                    │            │
             │  └─────────────┘   full text + rich metadata        │            │
             │  ┌─────────────┐◀──────────────────────────────────┘            │
             │  │  DynamoDB   │                                                  │
             │  │ (documents) │                                                  │
             │  └─────────────┘                                                  │
             │                                                                   │
  search  ──▶│  1) query_vectors → keys+distance   2) BatchGetItem → documents   │
             │  3) MMR rerank / RRF hybrid fusion → ranked SearchResults         │
             └───────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```python
from dynavec import Dynavec, DynavecConfig, Document
from dynavec.embeddings import OpenAIEmbedder   # or Gemini / Bedrock / SentenceTransformer

cfg = DynavecConfig(
    vector_bucket="my-vectors",     # S3 vector bucket
    index="docs",                   # vector index
    table="dynavec_docs",           # DynamoDB table
    dimension=1536,
    distance_metric="cosine",
    region="us-east-1",
    auto_provision=True,            # create bucket + index + table if missing
)

db = Dynavec(cfg, embedder=OpenAIEmbedder(model="text-embedding-3-small"))

db.upsert(
    [
        Document(id="a", text="The mitochondria is the powerhouse of the cell.",
                 metadata={"topic": "biology", "year": 2021}),
        Document(id="b", text="Rockets reach orbit at roughly 28,000 km/h.",
                 metadata={"topic": "space", "year": 2023}),
    ],
    auto_metadata=True,             # also attach hash/timestamp/word counts
)

hits = db.search(
    "how do cells make energy?",
    top_k=3,
    filter={"topic": "biology"},    # S3 Vectors metadata pre-filter
    rerank="mmr",                   # diversity-aware reranking
)
for h in hits:
    print(h.score, h.id, h.text)
```

### Bring your own vectors (no embedder needed)

```python
db = Dynavec(cfg)  # no embedder
db.upsert([Document(id="x", vector=my_1536_dim_vector, metadata={"lang": "en"})])
hits = db.search(vector=my_query_vector, top_k=5)
```

### The metadata switch

- **You provide metadata** → stored verbatim (full copy in DynamoDB, filterable subset in S3 Vectors).
- **`auto_metadata=True`** → dynavec also derives `created_at`, `content_hash`, `word_count`, `char_count`. Your keys always win on conflict.

Control the split with `DynavecConfig.filterable_keys` (allowlist of keys pushed to S3 Vectors for filtering) — keep it small; S3 Vectors caps filterable metadata size per vector.

---

## Framework integrations

### LangChain

```python
from dynavec.integrations.langchain import DynavecVectorStore

store = DynavecVectorStore(db, namespace="kb")
retriever = store.as_retriever(search_kwargs={"k": 4})
```

LlamaIndex, CrewAI, and Strands adapters are on the roadmap; the core client works in any of them today.

---

## Namespaces & multi-tenancy

Every write/read takes a `namespace`. dynavec tags each vector with its namespace and scopes queries to it automatically, so a single index can host many tenants (or many embedding "collections") with clean isolation. DynamoDB keys are `"{namespace}#{id}"` for even partition distribution.

---

## Provisioning & IAM

`auto_provision=True` (or `db.provision()`) creates the S3 vector bucket, the vector index, and the DynamoDB table idempotently. The caller needs `s3vectors:*` on the bucket/index and `dynamodb:*` on the table (scope these down in production — see [ARCHITECTURE.md](ARCHITECTURE.md)).

---

## Benchmarking

`benchmarks/` contains a harness that measures recall@k, latency (p50/p95/p99), and estimated \$/month against a labeled dataset, plus a cost model comparing dynavec to Pinecone / Qdrant / Milvus / Weaviate / OpenSearch. See [benchmarks/README.md](benchmarks/README.md).

```bash
pip install "dynavec[benchmark]"
python -m benchmarks.run_benchmark --dataset synthetic --n 10000 --dim 384
```

---

## Status

v0.1 — core hybrid client, pluggable embedders, RRF + MMR, provisioning, LangChain adapter, benchmark harness. **Roadmap:** async client, in-process `hnswlib` hot tier, sparse/BM25 hybrid via DynamoDB, LlamaIndex/CrewAI adapters.

## License

Apache-2.0
