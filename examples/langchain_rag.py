"""Use dynavec as a LangChain VectorStore / retriever.

    pip install "dynavec[sentence-transformers,langchain]"
    python examples/langchain_rag.py
"""

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.langchain import DynavecVectorStore

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="lc-rag",
    table="dynavec_lc_rag",
    dimension=embedder.dimension,
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)
store = DynavecVectorStore(db, namespace="kb")

store.add_texts(
    texts=[
        "LangChain is a framework for building LLM applications.",
        "Retrieval-augmented generation grounds answers in your own documents.",
        "dynavec keeps all vectors inside your own AWS account.",
    ],
    metadatas=[{"src": "docs"}, {"src": "docs"}, {"src": "readme"}],
    ids=["lc-1", "lc-2", "lc-3"],
)

retriever = store.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("where is my data stored?")
for d in docs:
    print(d.metadata.get("score"), "-", d.page_content)
