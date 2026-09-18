"""Multi-Query and HyDE retrieval -- runs offline, no AWS or LLM needed.

    python examples/query_expansion.py

The corpus below never uses the words in the user's query ("car won't start"),
which is the vocabulary-mismatch case where single-query search struggles. Both
retrievers take plain callables for the LLM step; here they are canned stand-ins.
In production swap in a real LLM, e.g.:

    def generate_queries(q: str) -> list[str]:
        reply = my_llm(f"Give 3 alternative search queries for: {q}. One per line.")
        return [line.strip() for line in reply.splitlines() if line.strip()]

    def generate_hypothetical(q: str) -> str:
        return my_llm(f"Write a short passage that answers: {q}")

    retriever = MultiQueryRetriever(db.namespace("kb"), generate_queries, top_k=3)
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
    """Toy lexical embedder: hashed bag of words (so it needs word overlap)."""

    dimension = DIM

    def embed_documents(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"[a-z']+", text.lower()):
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            out.append(vec)
        return out


# --- tiny in-memory stand-ins for S3 Vectors + DynamoDB (demo only) -----------
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
            return 1 - dot / norm

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


def show(title, hits):
    print(f"\n{title}")
    for h in hits:
        print(f"  {h.id:<9} {h.text[:70]}")


def main() -> None:
    cm.S3VectorsStore, cm.DynamoDBStore = _S3, _DDB  # demo only: no AWS
    cfg = DynavecConfig(vector_bucket="demo", index="demo", table="demo", dimension=DIM)
    db = Dynavec(cfg, embedder=BagOfWordsEmbedder())
    db.upsert([Document(id=k, text=v) for k, v in CORPUS.items()], namespace="kb")
    kb = db.namespace("kb")

    query = "car won't start"
    show("plain search (top 2):", kb.search(query, top_k=2))

    multi = MultiQueryRetriever(
        kb,
        generate_queries=lambda q: [
            "engine ignition failure starter motor",
            "dead battery spark plugs",
        ],
        top_k=2,
    )
    show("MultiQueryRetriever (top 2):", multi.search(query))

    hyde = HyDERetriever(
        kb,
        generate_hypothetical=lambda q: (
            "When a car will not start, the cause is usually a dead battery, "
            "a failing starter motor, or worn spark plugs in the engine ignition."
        ),
        top_k=2,
    )
    show("HyDERetriever (top 2):", hyde.search(query))


if __name__ == "__main__":
    main()
