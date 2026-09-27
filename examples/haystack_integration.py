"""Use dynavec as a Haystack DocumentStore and retriever.

    pip install "dynavec[sentence-transformers,haystack]"
    python examples/haystack_integration.py
"""

from haystack import Document

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.haystack import DynavecDocumentStore, DynavecRetriever

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="haystack-demo",
    table="dynavec_haystack",
    dimension=embedder.dimension,
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

store = DynavecDocumentStore(db, namespace="kb")

store.write_documents(
    [
        Document(
            id="hs-1",
            content="Haystack is a framework for building AI applications.",
            meta={"src": "docs"},
        ),
        Document(
            id="hs-2",
            content="Retrieval-augmented generation grounds answers in your documents.",
            meta={"src": "docs"},
        ),
        Document(
            id="hs-3",
            content="dynavec stores vectors inside your own AWS account.",
            meta={"src": "readme"},
        ),
    ]
)

retriever = DynavecRetriever(db, namespace="kb", top_k=2)

query = "where is my data stored?"
query_embedding = embedder.embed_query(query)

result = retriever.run(query_embedding=query_embedding)

for document in result["documents"]:
    print(document.id, "-", document.content)
