"""Wrap dynavec as a CrewAI tool so an agent can retrieve from it.

    pip install "dynavec[sentence-transformers,crewai]"
    python examples/crewai_tool.py
"""

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.tools import as_crewai_tool

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="crewai-rag",
    table="dynavec_crewai_rag",
    dimension=embedder.dimension,
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

db.upsert(
    [
        Document(id="cw-1", text="CrewAI orchestrates multiple agents that collaborate on a task."),
        Document(id="cw-2", text="A CrewAI tool is any callable an agent can invoke to fetch information."),
        Document(id="cw-3", text="dynavec keeps all vectors inside your own AWS account."),
    ],
    namespace="kb",
)

search_tool = as_crewai_tool(db, name="dynavec_search", namespace="kb", top_k=2)

# The tool behaves like any other CrewAI tool: give it to an Agent, or call it
# directly the way CrewAI itself does, via .run(...).
result = search_tool.run("what is a CrewAI tool?")
print(result)