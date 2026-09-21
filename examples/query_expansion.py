"""Multi-Query and HyDE retrieval -- runs offline, no AWS or LLM needed.

    python examples/query_expansion.py

The corpus below never uses the words in the user's query ("car won't start"),
which is the classic vocabulary-mismatch problem where single-query search fails.
Both retrievers take plain callables for the LLM step; here we use canned stand-ins.
In production, swap in any LLM (OpenAI, Anthropic, Ollama, LiteLLM):

    def generate_queries(q: str) -> list[str]:
        reply = my_llm(f"Give 3 alternative search queries for: {q}. One per line.")
        return [line.strip() for line in reply.splitlines() if line.strip()]

    def generate_hypothetical(q: str) -> str:
        return my_llm(f"Write a short passage that answers: {q}")

    retriever = db.namespace("kb").as_multiquery_retriever(generate_queries, top_k=3)
"""

from __future__ import annotations

import math
import re
import zlib

import dynavec.client as cm
from dynavec import Document, Dynavec, DynavecConfig, HyDERetriever, MultiQueryRetriever
from dynavec.embeddings.base import Embedder

DIM = 128


class BagOfWordsEmbedder(Embedder):
    """Toy lexical embedder: hashed bag of words (requires word overlap)."""

    dimension = DIM

    def embed_documents(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"[a-z']+", text.lower()):
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            out.append(vec)
        return out


# --- In-memory stand-ins for S3 Vectors + DynamoDB (demo only) ----------------
class _S3(cm.S3VectorsStore):
    def __init__(self, config, boto_session=None):
        self.config, self._store = config, {}

    def put_vectors(self, vectors):
        for key, vec, meta in vectors:
            self._store[key] = (list(vec), dict(meta))

    def query(self, query_vector, top_k, filter=None, **_):
        def dist(v):
            dot = sum(a * b for a, b in zip(query_vector, v))
            norm = (math.sqrt(sum(a * a for a in query_vector)) or 1e-9) * (
                math.sqrt(sum(b * b for b in v)) or 1e-9
            )
            return 1.0 - dot / norm

        rows = sorted(((k, dist(v)) for k, (v, _) in self._store.items()), key=lambda r: r[1])
        return [{"key": k, "distance": d} for k, d in rows[:top_k]]


class _DDB(cm.DynamoDBStore):
    def __init__(self, config, boto_session=None):
        self.config, self._store = config, {}

    def put_many(self, namespace, items):
        for doc_id, text, meta in items:
            self._store[(namespace, doc_id)] = {"text": text, "metadata": dict(meta)}

    def get_many(self, namespace, ids):
        return {i: self._store[(namespace, i)] for i in ids if (namespace, i) in self._store}


CORPUS = {
    "ignition": "Diagnosing engine ignition failure: check the battery, starter motor and spark plugs.",
    "brakes": "Replacing worn brake pads and rotors on a front disc brake assembly.",
    "tires": "Rotating tires and checking tread depth improves handling and fuel economy.",
    "oil": "Changing engine oil and the oil filter every five thousand miles.",
    "coolant": "Flushing the radiator coolant prevents the engine from overheating in summer.",
}


def show(title: str, hits: list) -> None:
    print(f"\n{title}")
    for rank, h in enumerate(hits, 1):
        print(f"  {rank}. [{h.id:<9}] {h.text[:65]}... (score: {h.score:.4f})")


def main() -> None:
    cm.S3VectorsStore, cm.DynamoDBStore = _S3, _DDB  # demo only: no AWS
    cfg = DynavecConfig(vector_bucket="demo", index="demo", table="demo", dimension=DIM)
    db = Dynavec(cfg, embedder=BagOfWordsEmbedder())
    db.upsert([Document(id=k, text=v) for k, v in CORPUS.items()], namespace="kb")
    kb = db.namespace("kb")

    query = "car won't start"
    print(f"User Query: {query!r}")
    print("Corpus documents never mention 'car' or 'start'. Notice how each method performs:")

    # 1. Plain search (misses the correct document)
    show("1. Plain search (top 2):", kb.search(query, top_k=2))

    # 2. MultiQueryRetriever (reformulates into technical terminology)
    multi = MultiQueryRetriever(
        kb,
        generate_queries=lambda q: [
            "engine ignition failure starter motor",
            "dead battery spark plugs",
        ],
        top_k=2,
    )
    show("2. MultiQueryRetriever (top 2):", multi.search(query))

    # 3. HyDERetriever (generates a hypothetical answer passage)
    hyde = kb.as_hyde_retriever(
        generate_hypothetical=lambda q: (
            "When an engine fails to crank, the issue is typically a depleted battery, "
            "a worn starter motor, or fouled spark plugs in the ignition system."
        ),
        top_k=2,
    )
    show("3. HyDERetriever - Single Passage (top 2):", hyde.search(query))

    # 4. HyDERetriever with multi-passage Centroid averaging
    hyde_multi = HyDERetriever(
        kb,
        generate_hypothetical=lambda q: [
            "A no-start condition is caused by faulty battery connections or a dead starter motor.",
            "Inspect the engine ignition system, spark plug wiring, and starter relay switch.",
        ],
        strategy="average",
        top_k=2,
    )
    show("4. HyDERetriever - Centroid Multi-Passage (top 2):", hyde_multi.search(query))


if __name__ == "__main__":
    main()
