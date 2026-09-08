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
        ("faq", "FAQ"),
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


# Reused in the Credentials page — keep in sync with docs/iam-policy.json.
POLICY_JSON = """{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DynavecS3Vectors",
      "Effect": "Allow",
      "Action": [
        "s3vectors:CreateVectorBucket", "s3vectors:GetVectorBucket",
        "s3vectors:ListVectorBuckets", "s3vectors:DeleteVectorBucket",
        "s3vectors:CreateIndex", "s3vectors:GetIndex",
        "s3vectors:ListIndexes", "s3vectors:DeleteIndex",
        "s3vectors:PutVectors", "s3vectors:GetVectors",
        "s3vectors:ListVectors", "s3vectors:QueryVectors", "s3vectors:DeleteVectors"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DynavecDynamoDB",
      "Effect": "Allow",
      "Action": [
        "dynamodb:CreateTable", "dynamodb:DescribeTable", "dynamodb:DeleteTable",
        "dynamodb:BatchWriteItem", "dynamodb:BatchGetItem",
        "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
        "dynamodb:DeleteItem", "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/dynavec_*",
        "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/dynavec_*/index/*"
      ]
    }
  ]
}"""

ENV_SAMPLE = """# .env  (keep this file private — never commit it)
AWS_ACCESS_KEY_ID=AKIA...your-key-id...
AWS_SECRET_ACCESS_KEY=...your-secret...
AWS_REGION=ap-south-1

OPENAI_API_KEY=sk-...        # or GOOGLE_API_KEY / COHERE_API_KEY"""


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
  <a href="faq.html"><h3>FAQ</h3><p>Common questions about regions, limits, consistency, and costs.</p></a>
  <a href="benchmarking.html"><h3>Benchmarking</h3><p>Recall, latency, and cost modeled across dimensions and scale.</p></a>
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
    top_k_page_size=50,           # optional client-side stream batch size
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
<tr><td><code>top_k_page_size</code></td><td>Client-side stream hydration batch. <code>None</code> (default) uses native S3 Vectors pages (at most 100). Does not change the service page size.</td></tr>
<tr><td><code>max_workers</code>, <code>parallel_writes</code></td><td>Thread-pool <a href="concurrency.html">concurrency</a> controls.</td></tr>
</table>
""")

PAGES["faq"] = ("Frequently Asked Questions",
    "Common questions about regions, limits, consistency, costs, and architecture.",
    """
<h2>Regions &amp; Availability</h2>

<h3>Which AWS regions are supported?</h3>
<p><code>dynavec</code> runs in any AWS region where both <strong>Amazon S3 Vectors</strong> and <strong>Amazon DynamoDB</strong> are supported (for example, <code>us-east-1</code>, <code>us-west-2</code>, <code>ap-south-1</code>, <code>eu-west-1</code>, and others). Specify your target region in <code>DynavecConfig(region="...")</code>.</p>

<h3>Can I query across multiple AWS regions?</h3>
<p>Each <code>DynavecConfig</code> connects to a single AWS region. Keeping your S3 vector bucket and DynamoDB table in the same region as your application workloads (e.g. Lambda, ECS, EC2) ensures single-digit millisecond latency and eliminates cross-region data transfer fees. If you need multi-region deployments, create independent <code>Dynavec</code> client instances per region.</p>

<h3>Can I use Amazon Bedrock or third-party embedders in different regions?</h3>
<p>Yes. <code>BedrockEmbedder</code> accepts an explicit <code>region</code> parameter (e.g. <code>BedrockEmbedder(model_id="amazon.titan-embed-text-v2:0", region="us-east-1")</code>) even if your vector database resources reside in a different region. Third-party embedders (OpenAI, Gemini, Cohere, Voyage) operate over public HTTPS APIs regardless of your AWS region.</p>

<h2>Limits &amp; Constraints</h2>

<h3>What is the maximum supported embedding dimension?</h3>
<p>Amazon S3 Vectors supports vectors up to <strong>4,096 dimensions</strong>. All popular embedding models fall well within this ceiling, including 384-d (MiniLM), 768-d (BGE-base, Gemini), 1024-d (Voyage, BGE-large), 1536-d (OpenAI small / ada-002), and 3072-d (OpenAI large). See <a href="https://github.com/codeforstartups/dynavec/blob/development/docs/EMBEDDING_DIMENSIONS.md">Embedding Dimensions Guide</a> for trade-offs.</p>

<h3>What are the limits on metadata filtering?</h3>
<p>dynavec uses a two-store hybrid model for metadata:</p>
<ul>
  <li><strong>Filterable metadata:</strong> Indexed directly in S3 Vectors for fast pre-filtering. Pass only the keys you need to filter on via <code>DynavecConfig(filterable_keys=[...])</code> (e.g. <code>["topic", "tenant_id", "year"]</code>).</li>
  <li><strong>Document metadata &amp; text:</strong> The complete payload is stored in DynamoDB, subject to DynamoDB's standard <strong>400 KB per item</strong> limit.</li>
</ul>

<h3>What is the maximum <code>top_k</code> query limit?</h3>
<p>Amazon S3 Vectors returns up to 100 vectors per page and supports querying up to the service ceiling of <strong>10,000 vectors</strong> per query via pagination. dynavec automatically handles <code>nextToken</code> pagination behind the scenes, and provides <a href="streaming.html"><code>search_stream()</code></a> to stream hydrated results progressively as each page arrives.</p>

<h3>What are the batch limits for ingestion and reads?</h3>
<p>DynamoDB processes up to 25 items per <code>BatchWriteItem</code> and 100 items per <code>BatchGetItem</code>. dynavec manages batch chunking, throttling retries, and parallel dispatch over a thread pool automatically — you can pass arbitrarily large lists of documents to <code>db.upsert()</code> or <code>ingest()</code>.</p>

<h2>Consistency &amp; Latency</h2>

<h3>Why do newly upserted vectors not show up immediately in search results?</h3>
<p>Amazon S3 Vectors is <strong>eventually consistent</strong> after ingestion. It typically takes a few seconds for new or modified vectors to be indexed and searchable via ANN vector queries. In contrast, document text and metadata written to DynamoDB are immediately accessible via key lookups.</p>
<div class="callout">When writing automated integration tests, insert a brief sleep (e.g. 3–5 seconds) after upsert before executing query assertions.</div>

<h3>What query latency should I expect?</h3>
<p>For end-to-end semantic searches (S3 Vectors ANN lookup + DynamoDB <code>BatchGetItem</code> document hydration):</p>
<ul>
  <li><strong>p50 latency:</strong> ~45 ms for warm queries.</li>
  <li><strong>p95 latency:</strong> ~120 ms.</li>
  <li><strong>Repeated queries:</strong> Sub-millisecond to low single-digit ms when using <a href="caching.html"><code>SemanticCache</code></a> or <a href="caching.html"><code>RedisCache</code></a>.</li>
</ul>

<h2>Costs &amp; Billing</h2>

<h3>How much does dynavec cost to run?</h3>
<p>dynavec has <strong>no baseline idle cost</strong> and no fixed monthly cluster fees. You only pay standard AWS pay-as-you-go rates:</p>
<table class="doc__params">
<tr><th>Component</th><th>Pricing Model</th></tr>
<tr><td><strong>S3 Vectors</strong></td><td>Vector storage (GB/month) + vector query and ingest PUT requests</td></tr>
<tr><td><strong>DynamoDB</strong></td><td>On-Demand Read/Write Request Units (RRUs/WRUs) + document storage</td></tr>
</table>
<p>For a typical workload with 1M vectors (768-d) and 1M queries/month, total AWS infrastructure cost is approximately <strong>$3–$4/month</strong> — up to 50–200× cheaper than running dedicated clusters (e.g. OpenSearch, Qdrant, Milvus).</p>

<h3>Are there data transfer fees between DynamoDB and S3 Vectors?</h3>
<p>No. When your application and dynavec resources are in the same AWS region, all data transfer between S3 Vectors, DynamoDB, and your compute environment (Lambda, ECS, EC2) is free.</p>

<h2>Security, Privacy &amp; Architecture</h2>

<h3>Does my data ever leave my AWS account?</h3>
<p>No. All documents, metadata, and vectors are stored inside your own AWS account's DynamoDB tables and S3 vector buckets. If you use <code>BedrockEmbedder</code> or <code>SentenceTransformerEmbedder</code>, embeddings are generated entirely within your AWS boundary or locally offline.</p>

<h3>How does multi-tenancy work?</h3>
<p>dynavec provides native <a href="namespaces.html">Namespaces</a>. A single S3 vector bucket and DynamoDB table can host many independent tenants. DynamoDB partition keys are cleanly isolated via escaped <code>"{namespace}#{id}"</code> prefixes, and vector queries are automatically scoped so data never leaks across namespaces.</p>
""")

PAGES["embeddings"] = ("Embeddings",
    "Pluggable, bring-your-own-key — or bring your own vectors.",
    """
<p>Choose an embedder and supply your own API key, or skip the embedder entirely and pass pre-computed vectors.
Embedder backends are imported lazily, so the base install stays light.</p>
""" + code("""from dynavec.embeddings import (
    OpenAIEmbedder, GeminiEmbedder, MistralEmbedder, BedrockEmbedder,
    SentenceTransformerEmbedder,
)

# hosted (BYO key via env var or argument)
emb = OpenAIEmbedder(model="text-embedding-3-small")        # 1536-d
emb = GeminiEmbedder(model="text-embedding-004")            # 768-d
emb = MistralEmbedder(model="mistral-embed")                # 1024-d

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
<p><code>search_stream()</code> is a generator: it yields one hit at a time as S3 Vectors paginates, so an agent can start
consuming the first results before the full set returns. Amazon S3 Vectors returns at most 100 vectors per response page;
dynavec follows <code>nextToken</code> up to the service limit of 10,000 results.</p>
""" + code("""for hit in db.search_stream("large query", top_k=250, namespace="kb", page_size=50):
    handle(hit)   # one hit at a time; page_size is the DynamoDB hydration batch
""") + """
<p><code>page_size</code> (or <code>DynavecConfig(top_k_page_size=50)</code>) only changes how many hits are hydrated from DynamoDB per batch. It does not change the S3 Vectors page size (fixed at 100) or time-to-first-result.</p>
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
<h2>Step-by-step: create the IAM user &amp; keys</h2>
<ol>
<li>AWS Console → <strong>IAM</strong> → <strong>Users</strong> → <strong>Create user</strong>. Name it <code>dynavec</code> (programmatic access only — no console sign-in needed).</li>
<li>On the permissions step choose <strong>Attach policies directly</strong>, then <strong>Create inline policy</strong> and open the <strong>JSON</strong> tab.</li>
<li>Paste the policy below, replacing <code>REGION</code> and <code>ACCOUNT_ID</code> with your region and 12-digit account id. Name it <code>dynavec-access</code> and create it.</li>
<li>Open the user → <strong>Security credentials</strong> → <strong>Create access key</strong> → <em>Application running outside AWS</em>. Copy the access key id and secret (shown once).</li>
<li>Put them in a <code>.env</code> file (below), then run any example.</li>
</ol>
""" + code(POLICY_JSON) + """
<div class="callout"><strong>Seeing red ARN errors in the JSON editor?</strong> Check the region spelling in the
DynamoDB ARNs — it must be a real region such as <code>ap-south-1</code>. A typo like <code>ap-soute-1</code>
makes the ARN invalid and shows two errors. The <code>s3vectors</code> block uses <code>"*"</code>, so it is
not affected.</div>
<h3>Your .env</h3>
""" + code(ENV_SAMPLE) + """
<p>Load it before running — <code>set -a &amp;&amp; . ./.env &amp;&amp; set +a</code> — or use
<code>python-dotenv</code>. dynavec then picks up the credentials automatically.</p>
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
""") + """
<h2>FastMCP server (Claude Desktop, Cursor, AI agents)</h2>
<p>Expose dynavec as an MCP server with <code>dynavec_search</code> and <code>dynavec_graph_search</code> tools. Configure via environment variables and launch over stdio:</p>
""" + code("""# Install with MCP extra
pip install "dynavec[mcp]"

# Launch the FastMCP server via CLI
dynavec mcp
""") + """
<p>Add to your Claude Desktop / Cursor configuration (<code>claude_desktop_config.json</code>):</p>
""" + code("""{
  "mcpServers": {
    "dynavec": {
      "command": "uvx",
      "args": ["--with", "dynavec[all]", "dynavec", "mcp"],
      "env": {
        "AWS_ACCESS_KEY_ID": "AKIA...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "AWS_REGION": "us-east-1",
        "OPENAI_API_KEY": "sk-...",
        "DYNAVEC_VECTOR_BUCKET": "my-vectors",
        "DYNAVEC_INDEX": "docs",
        "DYNAVEC_TABLE": "dynavec_docs"
      }
    }
  }
}
""") + """
<p>Programmatic initialization is also supported:</p>
""" + code("""from dynavec.mcp import create_mcp_server

mcp = create_mcp_server(db)
mcp.run(transport="stdio")
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
        side.append(f'<div class="side__group"><p class="side__title">{group}</p><ul class="side__list">')
        for s, label in items:
            cur = ' class="is-current"' if s == slug else ""
            side.append(f'<li><a href="{s}.html"{cur}>{label}</a></li>')
        side.append("</ul></div>")
    side.append("</nav>")

    # prev / next
    i = ORDER.index(slug)
    nxt = ""
    prev_a = next_a = ""
    if i > 0:
        p = ORDER[i - 1]
        prev_a = f'<a href="{p}.html"><span>Previous</span>{TITLES[p]}</a>'
    else:
        prev_a = "<span></span>"
    if i < len(ORDER) - 1:
        n = ORDER[i + 1]
        next_a = f'<a href="{n}.html" style="text-align:right"><span>Next</span>{TITLES[n]}</a>'
    nxt = f'<div class="doc__next">{prev_a}{next_a}</div>'

    crumbs = f'<div class="doc__crumbs"><a href="../index.html">dynavec</a> / <a href="index.html">Docs</a> / {title}</div>'

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
        with open(os.path.join(OUT, slug + ".html"), "w", encoding="utf-8") as f:
            f.write(render(slug))
    print(f"Wrote {len(PAGES)} docs pages to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
