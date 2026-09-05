"""Minimal end-to-end dynavec example (requires AWS credentials + S3 Vectors).

    pip install "dynavec[sentence-transformers]"
    python examples/quickstart.py
"""

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder

# Local, free, in-account embedding (no API key). 384-dim.
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="quickstart",
    table="dynavec_quickstart",
    dimension=embedder.dimension,
    distance_metric="cosine",
    region="us-east-1",
    auto_provision=True,   # creates bucket + index + table if missing
)

db = Dynavec(cfg, embedder=embedder)

db.upsert(
    [
        Document(id="1", text="The mitochondria is the powerhouse of the cell.",
                 metadata={"topic": "biology"}),
        Document(id="2", text="Rockets reach orbit at roughly 28,000 km/h.",
                 metadata={"topic": "space"}),
        Document(id="3", text="Photosynthesis converts sunlight into chemical energy.",
                 metadata={"topic": "biology"}),
    ],
    auto_metadata=True,
)

print("\n--- semantic search (no filter) ---")
for h in db.search("how do living things produce energy?", top_k=3):
    print(f"{h.score:.3f}  {h.id}  {h.text}")

print("\n--- filtered + MMR reranked ---")
for h in db.search("energy", top_k=2, filter={"topic": "biology"}, rerank="mmr"):
    print(f"{h.score:.3f}  {h.id}  {h.text}")
