"""Side-by-side retrieval test: dynavec vs Pinecone, same region, same vectors.

Embeds one corpus with OpenAI text-embedding-3-small (1536-d), indexes it in both
dynavec (S3 Vectors + DynamoDB) and Pinecone serverless, then runs the same
queries against both and reports top-k agreement + latency.

Prereqs
-------
    pip install "dynavec[openai]" pinecone     # or: uv add "dynavec[openai]" pinecone
    export OPENAI_API_KEY=sk-...
    export PINECONE_API_KEY=...
    export AWS_REGION=us-east-1                 # use the SAME region for both

Run
---
    python examples/pinecone_vs_dynavec.py
"""

from __future__ import annotations

import os
import time

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import OpenAIEmbedder

REGION = os.environ.get("AWS_REGION", "us-east-1")
DIM = 1536
TOP_K = 5

CORPUS = [
    ("d1", "The mitochondria is the powerhouse of the cell."),
    ("d2", "Photosynthesis converts sunlight into chemical energy."),
    ("d3", "Rockets reach low Earth orbit at roughly 28,000 km/h."),
    ("d4", "A black hole's event horizon is the point of no return."),
    ("d5", "Transformers use self-attention for long-range dependencies."),
    ("d6", "Vector databases power retrieval-augmented generation."),
    ("d7", "DynamoDB gives single-digit-millisecond key-value reads."),
    ("d8", "Amazon S3 Vectors is serverless approximate nearest-neighbor search."),
    ("d9", "Pinecone is a managed vector database service."),
    ("d10", "Cosine similarity measures the angle between two vectors."),
]
QUERIES = [
    "how do cells make energy?",
    "serverless vector search on AWS",
    "attention mechanism in deep learning",
]


def _overlap(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa))


def main() -> None:
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    ids = [i for i, _ in CORPUS]
    texts = [t for _, t in CORPUS]
    print(f"Embedding {len(texts)} docs with OpenAI (1536-d) ...")
    doc_vecs = embedder.embed_documents(texts)
    query_vecs = [embedder.embed_query(q) for q in QUERIES]

    # ---- dynavec ----
    print("Indexing in dynavec (S3 Vectors + DynamoDB) ...")
    db = Dynavec(
        DynavecConfig(
            vector_bucket="dynavec-vs-pinecone", index="cmp", table="dynavec_cmp",
            dimension=DIM, region=REGION, auto_provision=True,
        )
    )
    db.upsert([Document(id=i, text=t, vector=v) for (i, t), v in zip(CORPUS, doc_vecs)])

    # ---- pinecone ----
    print("Indexing in Pinecone (serverless) ...")
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx_name = "dynavec-cmp"
    existing = [i["name"] for i in pc.list_indexes()]
    if idx_name not in existing:
        pc.create_index(
            name=idx_name, dimension=DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=REGION),
        )
    pindex = pc.Index(idx_name)
    pindex.upsert(vectors=[{"id": i, "values": v} for i, v in zip(ids, doc_vecs)])

    print("Waiting for both indexes to settle ...")
    time.sleep(10)

    print(f"\n{'query':<40}{'agree@k':>10}{'dynavec ms':>13}{'pinecone ms':>14}")
    print("-" * 77)
    for q, qv in zip(QUERIES, query_vecs):
        t = time.perf_counter()
        dv = [h.id for h in db.search(vector=qv, top_k=TOP_K)]
        dv_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        pr = pindex.query(vector=qv, top_k=TOP_K)
        pr_ids = [m["id"] for m in pr["matches"]]
        pr_ms = (time.perf_counter() - t) * 1000

        print(f"{q[:38]:<40}{_overlap(dv, pr_ids):>10.2f}{dv_ms:>13.0f}{pr_ms:>14.0f}")

    db.close()
    print("\nNote: 'agree@k' is the fraction of dynavec's top-k also in Pinecone's "
          "top-k on identical embeddings — a practical apples-to-apples retrieval check.")


if __name__ == "__main__":
    main()
