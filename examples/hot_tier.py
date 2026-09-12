"""In-memory hot tier — Pinecone-class latency without a paid cluster.

S3 Vectors is cheap and serverless, but its per-query server time is hundreds of
milliseconds. For the hot working set, turn on the in-memory hot tier: after
``warm()``, a namespace is served entirely from RAM — no S3 Vectors query and no
DynamoDB hydration — dropping p50 from hundreds of ms to sub-millisecond.

    export OPENAI_API_KEY=...        # or use bring-your-own vectors
    python -m examples.hot_tier
"""

import time

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import OpenAIEmbedder

cfg = DynavecConfig(
    vector_bucket="dynavec-demo-vectors",
    index="docs",
    table="dynavec_demo",
    dimension=1536,
    region="us-east-1",
    auto_provision=True,
    # --- the hot tier ---
    hot_tier=True,               # keep a hot working set in RAM
    hot_tier_max_vectors=200_000,  # global RAM safety cap
)

db = Dynavec(cfg, embedder=OpenAIEmbedder(model="text-embedding-3-small"))

db.upsert(
    [
        Document(id="a", text="Mitochondria are the powerhouse of the cell.",
                 metadata={"topic": "biology"}),
        Document(id="b", text="Rockets reach orbit at roughly 28,000 km/h.",
                 metadata={"topic": "space"}),
        Document(id="c", text="Photosynthesis converts light into chemical energy.",
                 metadata={"topic": "biology"}),
    ]
)

# Make this namespace RAM-resident. Loads existing vectors from S3 Vectors,
# hydrates their text from DynamoDB, and marks the namespace authoritative.
loaded = db.warm(namespace="default")
print(f"warmed {loaded} vectors into RAM")
print("hot tier stats:", db.hot_stats())

# Served entirely from memory — filter, rerank, and scoring all run in-process.
t0 = time.perf_counter()
hits = db.search("how do cells make energy?", top_k=3, filter={"topic": "biology"})
dt = (time.perf_counter() - t0) * 1000
print(f"\nquery served in {dt:.2f} ms (from RAM):")
for h in hits:
    print(f"  {h.score:.3f}  {h.id}  {h.text}")

db.close()
