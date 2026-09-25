"""Use dynavec as a DSPy retrieval module.

    pip install "dynavec[sentence-transformers,dspy]"
    python examples/dspy_retriever.py
"""

import dspy

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.dspy import DynavecRM

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="dspy-rag",
    table="dynavec_dspy_rag",
    dimension=embedder.dimension,
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

db.upsert(
    [
        {
            "id": "dspy-1",
            "text": "DSPy is a framework for programming language model applications.",
            "metadata": {"source": "docs"},
        },
        {
            "id": "dspy-2",
            "text": "Retrieval-augmented generation grounds answers in retrieved documents.",
            "metadata": {"source": "docs"},
        },
        {
            "id": "dspy-3",
            "text": "Dynavec keeps vector data inside your own AWS account.",
            "metadata": {"source": "readme"},
        },
    ],
    namespace="kb",
)

rm = DynavecRM(db, namespace="kb", k=2)
dspy.configure(rm=rm)

retriever = dspy.Retrieve(k=2)
result = retriever("where is my data stored?")

for passage in result.passages:
    print(passage)
