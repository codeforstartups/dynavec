#!/usr/bin/env python3
"""Generate the dynavec docs site (one HTML page per feature) from this file.

Run:  python tools/build_docs.py
Outputs into opensource/dynavec/docs/. Pages share ../styles.css + docs.css/js.
"""

from __future__ import annotations

import os

OUT = os.path.join(os.path.dirname(__file__), "..", "opensource", "dynavec", "docs")

# Sidebar structure: (group title, [(slug, nav label)])
NAV = [
    ("Getting started", [
        ("index", "Overview"),
        ("installation", "Installation"),
        ("quickstart", "Quickstart"),
        ("configuration", "Configuration"),
    ]),
    ("Writing data", [
        ("embeddings", "Embeddings"),
        ("upsert", "Upsert"),
        ("update-and-lambda", "Update & Lambda"),
        ("ingestion", "Ingestion & MCP"),
    ]),
    ("Searching", [
        ("search", "Search"),
        ("metrics-and-rerank", "Metrics & rerank"),
        ("namespaces", "Namespaces"),
        ("streaming", "Streaming"),
        ("caching", "Caching"),
    ]),
    ("Advanced", [
        ("knowledge-graph", "Knowledge graph"),
        ("quantization", "Product quantization"),
        ("concurrency", "Concurrency"),
        ("credentials", "Credentials & IAM"),
    ]),
    ("Ecosystem", [
        ("integrations", "Framework integrations"),
        ("benchmarking", "Benchmarking"),
    ]),
]

# flat order for prev/next
ORDER = [slug for _, items in NAV for slug, _ in items]
TITLES = {slug: label for _, items in NAV for slug, label in items}


def code(src: str) -> str:
    return '<pre class="code"><code>' + src.strip("\n") + "</code></pre>"


# ---- page bodies (slug -> (title, subtitle, html)) ----
PAGES: dict[str, tuple[str, str, str]] = {}

PAGES["index"] = ("dynavec documentation",
    "A serverless hybrid vector database that runs inside your own AWS account.",
    """
<p>dynavec fuses <a href="https://aws.amazon.com/dynamodb/">Amazon DynamoDB</a> (single-digit-millisecond
metadata &amp; document store) with <a href="https://aws.amazon.com/s3/features/vectors/">Amazon S3 Vectors</a>
(billion-scale serverless ANN) into one Python client — a drop-in alternative to Pinecone, Qdrant, Milvus,
Weaviate, and OpenSearch that bills only when you use it.</p>
<div class="callout">New here? Start with <a href="installation.html">Installation</a> then the
<a href="quickstart.html">Quickstart</a>. Every feature has its own page in the sidebar.</div>
<h2>Explore by feature</h2>
<div class="docgrid">
  <a href="embeddings.html"><h3>Embeddings</h3><p>Pluggable, bring-your-own-key: OpenAI, Gemini, Cohere, Bedrock, local.</p></a>
  <a href="search.html"><h3>Search</h3><p>Semantic search with metadata pre-filtering and document hydration.</p></a>
  <a href="metrics-and-rerank.html"><h3>Metrics &amp; rerank</h3><p>Cosine, dot, euclidean, manhattan, weighted combos, and MMR.</p></a>
  <a href="namespaces.html"><h3>Namespaces</h3><p>Per-tenant / per-collection isolation on one index.</p></a>
  <a href="knowledge-graph.html"><h3>Knowledge graph</h3><p>Entity-relationship traversal that guides the vector search.</p></a>
  <a href="caching.html"><h3>Caching</h3><p>Semantic, DynamoDB-TTL, or Redis / ElastiCache query caches.</p></a>
  <a href="quantization.html"><h3>Product quantization</h3><p>Compress cached vectors up to 32× with ADC distance.</p></a>
  <a href="ingestion.html"><h3>Ingestion &amp; MCP</h3><p>Pull, chunk, embed from any source — including any MCP server.</p></a>
  <a href="credentials.html"><h3>Credentials &amp; IAM</h3><p>Access keys, profiles, cross-account assume-role, least-privilege policy.</p></a>
  <a href="integrations.html"><h3>Integrations</h3><p>LangChain, LlamaIndex, and a tool for LangGraph / CrewAI / Strands.</p></a>
</div>
""")

PAGES["installation"] = ("Installation",
    "Base install is boto3 + numpy. Everything else is an optional extra.",
    """
<p>Install with pip or uv — both pull from PyPI.</p>
""" + code("""# base (boto3 + numpy only)
pip install dynavec
uv add dynavec

# with an embedder
pip install "dynavec[openai]"
pip install "dynavec[sentence-transformers]"

# everything (all embedders + framework adapters)
pip install "dynavec[all]"
""") + """
<h2>Optional extras</h2>
<table class="doc__params">
<tr><th>Extra</th><th>Adds</th></tr>
<tr><td><code>openai</code>, <code>gemini</code>, <code>cohere</code></td><td>hosted embedders (bring your own key)</td></tr>
<tr><td><code>sentence-transformers</code></td><td>local / offline embedder, also used for cross-encoder rerank</td></tr>
<tr><td><code>langchain</code>, <code>llamaindex</code>, <code>crewai</code></td><td>framework vector stores / tools</td></tr>
<tr><td><code>redis</code></td><td>RedisCache (AWS ElastiCache)</td></tr>
<tr><td><code>mcp</code></td><td>MCP client ingestion</td></tr>
<tr><td><code>benchmark</code></td><td>pandas / matplotlib for the benchmark suite</td></tr>
</table>
<div class="callout">dynavec needs AWS credentials with permission for S3 Vectors and DynamoDB.
See <a href="credentials.html">Credentials &amp; IAM</a>.</div>
""")

PAGES["quickstart"] = ("Quickstart",
    "Provision, upsert, and search in a dozen lines.",
    """
<p>With <code>auto_provision=True</code>, dynavec creates the S3 vector bucket, the vector index, and the
DynamoDB table on first use.</p>
""" + code("""from dynavec import Dynavec, DynavecConfig, Document
from dynavec.embeddings import OpenAIEmbedder

cfg = DynavecConfig(
    vector_bucket="my-vectors",
    index="docs",
    table="dynavec_docs",
    dimension=1536,
    region="us-east-1",
    auto_provision=True,
)
db = Dynavec(cfg, embedder=OpenAIEmbedder(model="text-embedding-3-small"))

db.upsert([
    Document(id="a", text="Mitochondria power the cell.", metadata={"topic": "bio"}),
    Document(id="b", text="Rockets reach orbit near 28,000 km/h.", metadata={"topic": "space"}),
], auto_metadata=True)

for hit in db.search("how do cells make energy?", top_k=3):
    print(hit.score, hit.id, hit.text)
""") + """
<div class="callout">S3 Vectors is eventually consistent right after ingest — allow a few seconds before
querying freshly written vectors.</div>
""")

PAGES["configuration"] = ("Configuration",
    "Everything the client needs, in one frozen dataclass.",
    """
<p><code>DynavecConfig</code> is an immutable description of your resources and tuning. It never holds secrets —
credentials are passed separately (see <a href="credentials.html">Credentials &amp; IAM</a>).</p>
""" + code("""from dynavec import DynavecConfig

cfg = DynavecConfig(
    vector_bucket="my-vectors",   # S3 vector bucket name
    index="docs",                 # vector index name
    table="dynavec_docs",         # DynamoDB table name
    dimension=1536,               # must match your embedder
    distance_metric="cosine",     # "cosine" or "euclidean" (S3 Vectors native)
    region="us-east-1",
    filterable_keys=["topic"],    # metadata keys pushed to S3 Vectors for filtering
    over_fetch=4,                 # candidate multiplier when reranking
    max_workers=8,                # thread pool for parallel I/O
    auto_provision=True,
)
""") + """
<h2>Key parameters</h2>
<table class="doc__params">
<tr><th>Field</th><th>Meaning</th></tr>
<tr><td><code>dimension</code></td><td>Embedding size; must match the embedder and the index.</td></tr>
<tr><td><code>distance_metric</code></td><td>The S3 Vectors index metric. Other metrics are available at rerank time — see <a href="metrics-and-rerank.html">Metrics</a>.</td></tr>
<tr><td><code>filterable_keys</code></td><td>Small allowlist of metadata pushed to S3 Vectors. Everything else lives only in DynamoDB. Keep it small.</td></tr>
<tr><td><code>over_fetch</code></td><td>How many extra candidates to pull before reranking/rescoring.</td></tr>
<tr><td><code>max_workers</code>, <code>parallel_writes</code></td><td>Thread-pool <a href="concurrency.html">concurrency</a> controls.</td></tr>
</table>
""")

PAGES["embeddings"] = ("Embeddings",
    "Pluggable, bring-your-own-key — or bring your own vectors.",
    """
<p>Choose an embedder and supply your own API key, or skip the embedder entirely and pass pre-computed vectors.
Embedder backends are imported lazily, so the base install stays light.</p>
""" + code("""from dynavec.embeddings import (
    OpenAIEmbedder, GeminiEmbedder, BedrockEmbedder, SentenceTransformerEmbedder,
)

# hosted (BYO key via env var or argument)
emb = OpenAIEmbedder(model="text-embedding-3-small")        # 1536-d
emb = GeminiEmbedder(model="text-embedding-004")            # 768-d

# in-account (no third party) or fully local / offline
emb = BedrockEmbedder(model_id="amazon.titan-embed-text-v2:0", region="us-east-1")
emb = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2") # 384-d, free
""") + """
<h2>Bring your own vectors</h2>
<p>No embedder needed — pass vectors directly and query with a vector.</p>
""" + code("""from dynavec import Dynavec, DynavecConfig, Document

db = Dynavec(cfg)  # no embedder
db.upsert([Document(id="x", vector=my_1536d_vector, metadata={"lang": "en"})])
db.search(vector=my_query_vector, top_k=5)
""") + """
<div class="callout"><strong>Compliance tip:</strong> use <code>BedrockEmbedder</code> or
<code>SentenceTransformerEmbedder</code> to keep embedding in-account or offline — no data leaves your
environment.</div>
""")

PAGES["upsert"] = ("Upsert",
    "Write documents to both stores in one call.",
    """
<p>Each document carries an <code>id</code>, either <code>text</code> (which gets embedded) or a
<code>vector</code>, and optional <code>metadata</code>. dynavec splits metadata: a small filterable subset
goes to S3 Vectors, the full copy plus text goes to DynamoDB.</p>
""" + code("""from dynavec import Document

db.upsert([
    Document(id="1", text="apple pie recipe", metadata={"cat": "food", "rating": 5}),
    Document(id="2", text="rocket launch schedule", metadata={"cat": "space"}),
], namespace="kb", auto_metadata=True)
""") + """
<h2>The metadata switch</h2>
<ul>
<li><strong>You provide metadata</strong> — stored verbatim.</li>
<li><strong><code>auto_metadata=True</code></strong> — dynavec also derives <code>created_at</code>,
<code>content_hash</code>, <code>word_count</code>, <code>char_count</code>. Your keys win on conflict.</li>
</ul>
<p>Re-upserting the same <code>id</code> overwrites it. Writes to the two stores run in parallel; see
<a href="concurrency.html">Concurrency</a>. To change part of a document, use
<a href="update-and-lambda.html">update</a>.</p>
""")

PAGES["update-and-lambda"] = ("Update &amp; Lambda transforms",
    "Change text, vector, or metadata — and transform data in-account.",
    """
<p><code>update()</code> is a read-modify-write: metadata merges by default, and the vector is only
re-derived when the text changes or you pass a new vector.</p>
""" + code("""# merge new metadata, keep existing text + vector
db.update("1", namespace="kb", metadata={"rating": 4})

# change text -> re-embbeds and overwrites the vector
db.update("1", namespace="kb", text="new content")
""") + """
<h2>Transform pipeline</h2>
<p>Transforms are plain callables run on each document before it is written — for enrichment, redaction, or
deriving vectors elsewhere.</p>
""" + code("""from dynavec.transforms import TransformPipeline

def redact(ctx):
    ctx.metadata["pii"] = False
    return ctx

db.upsert(docs, transform=redact)             # or transform=TransformPipeline([...])
""") + """
<h2>Run the transform in your own AWS Lambda</h2>
<p><code>LambdaTransform</code> invokes a Lambda you own with the document payload and applies whatever it
returns — keeping custom logic in-account.</p>
""" + code("""from dynavec.transforms import LambdaTransform

xform = LambdaTransform("my-transform-fn", session=db._session)
db.upsert(docs, transform=xform)
""") + """
<div class="callout">Grant <code>lambda:InvokeFunction</code> on that function — see
<a href="credentials.html">Credentials &amp; IAM</a>.</div>
""")

PAGES["ingestion"] = ("Ingestion &amp; MCP",
    "Suck in content from anywhere — including any MCP server.",
    """
<p>A <em>source</em> is any iterable of records. <code>ingest()</code> chunks, embeds, and upserts them.</p>
""" + code("""from dynavec.ingest import ingest, IterableSource

src = IterableSource([
    {"id": "doc1", "text": long_text, "metadata": {"src": "wiki"}},
])
ingest(db, src, namespace="kb", chunk_size=1000, overlap=150)
""") + """
<h2>From any MCP server</h2>
<p><code>MCPResourceSource</code> turns an MCP server's <em>resources</em> (Notion, Confluence, Drive, your
own) into an embeddable corpus — no per-source code.</p>
""" + code("""from dynavec.ingest import ingest, MCPResourceSource

ingest(db, MCPResourceSource(mcp_session), namespace="kb")
""") + """
<p>Chunk ids are <code>"{record_id}#chunk{n}"</code> with <code>source_id</code> / <code>chunk</code>
metadata, so you can group or delete a whole document later.</p>
""")

PAGES["search"] = ("Search",
    "ANN in S3 Vectors, document hydration from DynamoDB.",
    """
<p>Provide a <code>query</code> string (embedded for you) or a raw <code>vector</code>. S3 Vectors returns the
nearest keys; dynavec hydrates the full documents from DynamoDB via <code>BatchGetItem</code>.</p>
""" + code("""hits = db.search(
    "how do cells make energy?",
    top_k=5,
    namespace="kb",
    filter={"topic": "biology"},   # metadata pre-filter (S3 Vectors)
)
for h in hits:
    print(h.score, h.id, h.text, h.metadata)
""") + """
<h2>Metadata filtering</h2>
<p>Filters use the S3 Vectors dialect — bare <code>{"k": v}</code> is equality; operators like
<code>$gte</code>, <code>$in</code>, <code>$and</code>, <code>$or</code> are supported. Only keys in
<code>filterable_keys</code> can be filtered.</p>
""" + code("""db.search("q", filter={"$and": [{"topic": "bio"}, {"year": {"$gte": 2020}}]})
""") + """
<p>Results come back as <code>SearchResult</code> with <code>id</code>, <code>score</code> (higher = more
similar), <code>distance</code>, <code>text</code>, and <code>metadata</code>. Refine ordering with
<a href="metrics-and-rerank.html">metrics &amp; rerank</a>, speed up repeats with
<a href="caching.html">caching</a>.</p>
""")

PAGES["metrics-and-rerank"] = ("Metrics &amp; rerank",
    "Cosine, dot, euclidean, manhattan, weighted combinations, and MMR.",
    """
<p>The S3 Vectors index metric is cosine or euclidean. On top of the returned candidates, dynavec can
<strong>rescore</strong> with any metric — or a weighted combination — client-side.</p>
""" + code("""# single metric
db.search("q", top_k=5, rescore="manhattan")

# weighted combination (normalized per candidate set)
db.search("q", top_k=5, rescore={"cosine": 0.7, "dot": 0.3})
""") + """
<h2>MMR diversity rerank</h2>
<p>Maximal Marginal Relevance balances relevance against diversity so results are not near-duplicates.</p>
""" + code("""db.search("q", top_k=5, rerank="mmr", mmr_lambda=0.5)  # 1=relevance, 0=diversity
""") + """
<p>Both over-fetch <code>top_k * over_fetch</code> candidates first. You can also fuse multiple result lists
with Reciprocal Rank Fusion:</p>
""" + code("""from dynavec import reciprocal_rank_fusion
fused = reciprocal_rank_fusion([dense_hits, keyword_hits])
"""))

PAGES["namespaces"] = ("Namespaces",
    "Multi-tenant / multi-collection isolation on a single index.",
    """
<p>Every operation takes a <code>namespace</code>. dynavec tags each vector with it and scopes queries
automatically, so one index can host many tenants. DynamoDB keys are <code>"{namespace}#{id}"</code> for even
partition distribution.</p>
""" + code("""kb = db.namespace("tenant-42")     # a view bound to one namespace
kb.upsert([Document(id="1", text="private doc")])
kb.search("scoped to this tenant only", top_k=4)
kb.delete(["1"])
""") + """
<div class="callout">Namespaces are the recommended way to do per-customer RAG: same infrastructure, clean
data isolation, no cross-tenant leakage.</div>
""")

PAGES["streaming"] = ("Streaming",
    "Deliver results to agents page-by-page as they arrive.",
    """
<p><code>search_stream()</code> is a generator: it yields hits as S3 Vectors paginates, so an agent can start
consuming the first results before the full set returns.</p>
""" + code("""for hit in db.search_stream("large query", top_k=100, namespace="kb"):
    handle(hit)   # arrives page by page
""") + """
<div class="callout">Reranking and rescoring need the full candidate set, so they are not applied in
streaming mode. Use <a href="search.html">search()</a> when you need them.</div>
""")

PAGES["caching"] = ("Caching",
    "Skip the vector DB for repeated or similar queries.",
    """
<p>Attach a cache and repeated queries are served without hitting S3 Vectors. Three backends:</p>
""" + code("""from dynavec import Dynavec, SemanticCache, DynamoDBCache, RedisCache

# 1) in-process semantic cache — also serves near-duplicate queries
db = Dynavec(cfg, embedder=emb, cache=SemanticCache(threshold=0.97))

# 2) durable, shared cache in your own DynamoDB table (TTL expiry)
db = Dynavec(cfg, embedder=emb, cache=DynamoDBCache(cfg, ttl_seconds=3600))

# 3) sub-millisecond shared cache on Redis / AWS ElastiCache
db = Dynavec(cfg, embedder=emb, cache=RedisCache("redis://my-elasticache:6379/0"))
""") + """
<table class="doc__params">
<tr><th>Backend</th><th>Best for</th></tr>
<tr><td><code>SemanticCache</code></td><td>single process; tolerant of near-duplicate hits; zero infra</td></tr>
<tr><td><code>DynamoDBCache</code></td><td>durable, shared, no extra service; exact-match with TTL</td></tr>
<tr><td><code>RedisCache</code></td><td>many workers/hosts; lowest latency; AWS ElastiCache</td></tr>
</table>
<p>Force a fresh search per call with <code>db.search(..., use_cache=False)</code>.</p>
""")

PAGES["knowledge-graph"] = ("Knowledge graph",
    "Attach meaning to embeddings and traverse it to guide search.",
    """
<p>Alongside the vector index, dynavec keeps a lightweight entity-relationship graph in DynamoDB. Entities link
to documents; you can traverse the graph first (cheap key lookups) to gather a candidate set, then rank only
those against the query embedding. That is the DynamoDB → S3 Vectors reference join.</p>
""" + code("""# build the graph
db.graph_add_edge("acme", "competes_with", "globex", namespace="kb")
db.graph_link("acme", ["doc-1", "doc-2"], namespace="kb")

# GraphRAG: traverse from seeds, then rank related docs by the query
hits = db.graph_search(
    "recent product launches",
    seed_entities=["acme"],
    hops=2,
    top_k=10,
    namespace="kb",
)
""") + """
<p>Traversal helpers: <code>graph_add_node</code>, <code>graph_add_edge</code>, <code>graph_link</code>,
<code>graph_neighbors</code>.</p>
<div class="callout">The graph uses embedded adjacency lists (one item per node). Very high fan-out entities
want a sort-key adjacency design — on the roadmap.</div>
""")

PAGES["quantization"] = ("Product quantization",
    "Compress cached vectors up to 32× with asymmetric distance.",
    """
<p>S3 Vectors stores float32 and manages its own layout, so PQ does not change what it stores. PQ compresses
the vectors <em>dynavec</em> caches — the in-memory hot tier and local candidate caches — turning a
<code>dim × 4</code> byte vector into <code>m</code> bytes.</p>
""" + code("""from dynavec import ProductQuantizer

pq = ProductQuantizer(m=96, nbits=8).fit(training_vectors)   # 768-d -> 96 bytes (32x)
codes = pq.encode(vectors)          # uint8 codes
dists = pq.asymmetric_distances(query, codes)   # ADC, fast at scale
print(pq.reconstruction_error(vectors))
""") + """
<table class="doc__params">
<tr><th>Param</th><th>Meaning</th></tr>
<tr><td><code>m</code></td><td>Number of subspaces; must divide the vector dimension.</td></tr>
<tr><td><code>nbits</code></td><td>Bits per subquantizer (8 → 256 centroids, uint8 codes).</td></tr>
</table>
""")

PAGES["concurrency"] = ("Concurrency",
    "GIL-aware thread pool for I/O-bound AWS calls.",
    """
<p>dynavec's workload is I/O-bound (network calls to AWS), and Python releases the GIL during those calls — so
a thread pool gives real parallelism without an async rewrite. Batched writes fan out across threads, and
<code>search_many</code> runs several queries concurrently.</p>
""" + code("""# many queries at once
results = db.search_many(["q1", "q2", "q3"], top_k=5, namespace="kb")

# tune the pool
cfg = DynavecConfig(..., max_workers=16, parallel_writes=True)

# clean up the pool (or use the client as a context manager)
with Dynavec(cfg, embedder=emb) as db:
    ...
""") + """
<div class="callout">A native asyncio client (<code>aioboto3</code>) is on the roadmap for very high
concurrency.</div>
""")

PAGES["credentials"] = ("Credentials &amp; IAM",
    "Connect to your account, with least-privilege permissions.",
    """
<p>dynavec uses the standard boto3 credential chain, so exported env vars just work. You can also pass keys
explicitly or assume a cross-account role.</p>
""" + code("""from dynavec import Dynavec, AWSCredentials

# explicit keys / profile / cross-account role
creds = AWSCredentials(
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1",
    # profile_name="prod",
    # assume_role_arn="arn:aws:iam::OTHER_ACCOUNT:role/dynavec",
)
db = Dynavec(cfg, credentials=creds)
""") + """
<h2>Least-privilege IAM policy</h2>
<p>The client needs S3 Vectors (buckets, indexes, vectors) and DynamoDB (table + item ops). A ready policy
lives at <a href="https://github.com/codeforstartups/dynavec/blob/development/docs/iam-policy.json">docs/iam-policy.json</a>.</p>
<table class="doc__params">
<tr><th>Service</th><th>Actions</th></tr>
<tr><td>s3vectors</td><td>Create/Get/List/Delete VectorBucket &amp; Index; Put/Get/List/Query/Delete Vectors</td></tr>
<tr><td>dynamodb</td><td>CreateTable, DescribeTable, Batch/Get/Put/Update/Delete Item, Query</td></tr>
<tr><td>bedrock <em>(optional)</em></td><td>InvokeModel — only for BedrockEmbedder</td></tr>
<tr><td>lambda <em>(optional)</em></td><td>InvokeFunction — only for LambdaTransform</td></tr>
</table>
<div class="callout"><strong>Never commit secrets.</strong> Use <code>.env</code> (gitignored) locally,
GitHub Secrets in CI, and prefer an IAM role over long-lived keys in production.</div>
""")

PAGES["integrations"] = ("Framework integrations",
    "LangChain, LlamaIndex, and a tool for any agent framework.",
    """
<h2>LangChain</h2>
""" + code("""from dynavec.integrations.langchain import DynavecVectorStore
store = DynavecVectorStore(db, namespace="kb")
retriever = store.as_retriever(search_kwargs={"k": 4})
""") + """
<h2>LlamaIndex</h2>
""" + code("""from dynavec.integrations.llamaindex import DynavecLlamaStore
from llama_index.core import VectorStoreIndex, StorageContext

store = DynavecLlamaStore(db, namespace="kb")
ctx = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(docs, storage_context=ctx)
""") + """
<h2>LangGraph / CrewAI / Strands</h2>
<p>A framework-agnostic retriever tool — just a callable that takes a query and returns text.</p>
""" + code("""from dynavec.integrations.tools import make_retriever_fn
retrieve = make_retriever_fn(db, top_k=4)   # fn(query: str) -> str
# also: as_langchain_tool(db), as_crewai_tool(db)
"""))

PAGES["benchmarking"] = ("Benchmarking",
    "Recall, latency, and cost — with tables and charts.",
    """
<p>The suite measures recall@k and latency against a labeled dataset and models cost versus Pinecone,
OpenSearch, Qdrant, Weaviate, and Milvus across dimensions and scale.</p>
""" + code("""pip install "dynavec[benchmark]"

# recall + latency (local control, or real AWS)
python -m benchmarks.run_benchmark --backend local --n 50000 --dim 384
python -m benchmarks.run_benchmark --backend dynavec --bucket my-vectors \\
    --index bench --table dynavec_bench --n 100000 --dim 1536

# cost comparison + charts (dimensions 384-3072, 100K -> 1B vectors)
python -m benchmarks.report --qpm 1_000_000
""") + """
<div class="callout">Cost figures are cost-model estimates from public list prices. Competitor recall/latency
are representative until you run the live benchmark against your own account.</div>
""")


def render(slug: str) -> str:
    title, sub, body = PAGES[slug]
    # sidebar
    side = ['<button class="side__toggle">☰ Menu</button>', '<nav class="side" aria-label="Docs">']
    for group, items in NAV:
        side.append('<div class="side__group"><p class="side__title">%s</p><ul class="side__list">' % group)
        for s, label in items:
            cur = ' class="is-current"' if s == slug else ""
            side.append('<li><a href="%s.html"%s>%s</a></li>' % (s, cur, label))
        side.append("</ul></div>")
    side.append("</nav>")

    # prev / next
    i = ORDER.index(slug)
    nxt = ""
    prev_a = next_a = ""
    if i > 0:
        p = ORDER[i - 1]
        prev_a = '<a href="%s.html"><span>Previous</span>%s</a>' % (p, TITLES[p])
    else:
        prev_a = "<span></span>"
    if i < len(ORDER) - 1:
        n = ORDER[i + 1]
        next_a = '<a href="%s.html" style="text-align:right"><span>Next</span>%s</a>' % (n, TITLES[n])
    nxt = '<div class="doc__next">%s%s</div>' % (prev_a, next_a)

    crumbs = '<div class="doc__crumbs"><a href="../index.html">dynavec</a> / <a href="index.html">Docs</a> / %s</div>' % title

    return TEMPLATE % {
        "title": title.replace("&amp;", "&"),
        "sub": sub,
        "side": "\n".join(side),
        "crumbs": crumbs if slug != "index" else '<div class="doc__crumbs"><a href="../index.html">dynavec</a> / Docs</div>',
        "body": body,
        "next": nxt,
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>%(title)s · dynavec docs</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="../styles.css" />
  <link rel="stylesheet" href="docs.css" />
</head>
<body>
  <header class="nav">
    <div class="wrap nav__inner">
      <a class="brand" href="../index.html">
        <svg class="brand__mark" width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
          <rect x="1" y="1" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <line x1="1" y1="11" x2="21" y2="11" stroke="currentColor" stroke-width="1.5"/>
          <line x1="11" y1="1" x2="11" y2="21" stroke="currentColor" stroke-width="1.5"/>
          <circle cx="6" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="16" r="2" fill="currentColor"/>
        </svg>
        <span class="brand__name">dynavec</span>
      </a>
      <nav class="nav__links" aria-label="Primary">
        <a href="../index.html#why">Why</a>
        <a href="../index.html#benchmarks">Benchmarks</a>
        <a href="index.html">Docs</a>
      </nav>
      <a class="btn btn--ghost star" href="https://github.com/codeforstartups/dynavec" target="_blank" rel="noopener">
        <span aria-hidden="true">&#9733;</span> Star <span class="star__count" data-stars>&mdash;</span>
      </a>
    </div>
  </header>

  <div class="docshell">
    %(side)s
    <main class="doc">
      %(crumbs)s
      <h1>%(title)s</h1>
      <p class="doc__sub">%(sub)s</p>
      %(body)s
      %(next)s
    </main>
  </div>

  <script src="docs.js"></script>
</body>
</html>
"""


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for slug in PAGES:
        with open(os.path.join(OUT, slug + ".html"), "w") as f:
            f.write(render(slug))
    print("Wrote %d docs pages to %s" % (len(PAGES), os.path.normpath(OUT)))


if __name__ == "__main__":
    main()
