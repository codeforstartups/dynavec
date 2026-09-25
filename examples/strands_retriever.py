"""Use dynavec as a Strands Agents retrieval tool.

pip install "dynavec[sentence-transformers]" strands-agents
python examples/strands_retriever.py
"""

from strands import Agent, tool

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.tools import make_retriever_fn

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="strands-rag",
    table="dynavec_strands_rag",
    dimension=embedder.dimension,
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

db.upsert(
    [
        Document(id="strands-1", text="Strands Agents builds agents from models and tools."),
        Document(
            id="strands-2", text="Retrieval-augmented generation grounds answers in documents."
        ),
        Document(id="strands-3", text="dynavec keeps vector data inside your own AWS account."),
    ],
    namespace="kb",
)

retrieve = make_retriever_fn(db, namespace="kb", top_k=2)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the dynavec knowledge base for relevant passages.

    Args:
        query: Natural-language question to search for.
    """

    return retrieve(query)


agent = Agent(tools=[search_knowledge_base])
response = agent("Where is dynavec vector data stored?")
print(response)
