"""Real end-to-end retrieval with OpenAI text-embedding-3-small (1536-dim).

Runs against YOUR AWS account and YOUR OpenAI key — dynavec auto-provisions the
S3 vector bucket, the vector index, and the DynamoDB table on first use.

Prereqs
-------
    pip install "dynavec[openai]"          # or: uv add "dynavec[openai]"
    export OPENAI_API_KEY=sk-...
    # AWS creds via env / profile / role, in a region where S3 Vectors is available
    export AWS_REGION=us-east-1

Run
---
    python examples/openai_1536_retrieval.py
"""

from __future__ import annotations

import os
import time

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import OpenAIEmbedder

REGION = os.environ.get("AWS_REGION", "us-east-1")

CORPUS = [
    ("d1", "The mitochondria is the powerhouse of the cell.", {"topic": "biology"}),
    ("d2", "Photosynthesis converts sunlight into chemical energy in plants.", {"topic": "biology"}),
    ("d3", "Rockets must reach ~28,000 km/h to achieve low Earth orbit.", {"topic": "space"}),
    ("d4", "A black hole's event horizon is the point of no return.", {"topic": "space"}),
    ("d5", "Transformers use self-attention to model long-range dependencies.", {"topic": "ai"}),
    ("d6", "Vector databases power retrieval-augmented generation for LLMs.", {"topic": "ai"}),
    ("d7", "DynamoDB offers single-digit-millisecond key-value lookups.", {"topic": "aws"}),
    ("d8", "Amazon S3 Vectors provides serverless approximate nearest-neighbor search.", {"topic": "aws"}),
]

QUERIES = [
    "how do cells produce energy?",
    "what does it take to get to orbit?",
    "serverless vector search on AWS",
]


def main() -> None:
    embedder = OpenAIEmbedder(model="text-embedding-3-small")  # 1536 dims
    assert embedder.dimension == 1536

    cfg = DynavecConfig(
        vector_bucket="dynavec-openai-demo",
        index="kb-1536",
        table="dynavec_openai_demo",
        dimension=1536,
        distance_metric="cosine",
        region=REGION,
        auto_provision=True,   # creates bucket + index + table if missing
    )

    print(f"Connecting to AWS ({REGION}) and provisioning resources ...")
    db = Dynavec(cfg, embedder=embedder)

    print(f"Embedding + upserting {len(CORPUS)} documents ...")
    t0 = time.perf_counter()
    db.upsert(
        [Document(id=i, text=t, metadata=m) for i, t, m in CORPUS],
        auto_metadata=True,
    )
    print(f"  upsert took {time.perf_counter() - t0:.2f}s")

    # S3 Vectors indexes are eventually consistent right after ingest.
    print("Waiting a few seconds for the index to become queryable ...")
    time.sleep(5)

    for q in QUERIES:
        print(f"\nQ: {q}")
        t = time.perf_counter()
        hits = db.search(q, top_k=3)
        dt = (time.perf_counter() - t) * 1000
        for h in hits:
            topic = h.metadata.get("topic", "?")
            print(f"  {h.score:.3f}  [{topic:<7}] {h.id}: {h.text}")
        print(f"  ({dt:.0f} ms round-trip)")

    print("\nFiltered query (topic=aws) with MMR rerank:")
    for h in db.search("fast AWS storage", top_k=2, filter={"topic": "aws"}, rerank="mmr"):
        print(f"  {h.score:.3f}  {h.id}: {h.text}")

    db.close()
    print("\nDone. Resources remain in your account (delete via verify_provisioning.py --cleanup).")


if __name__ == "__main__":
    main()
