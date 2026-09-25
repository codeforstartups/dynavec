"""End-to-end multi-tenant RAG with dynavec namespaces.

    pip install "dynavec[sentence-transformers]"
    python examples/multi_tenant_rag.py

Each tenant gets a namespace on the same Dynavec index. Retrieval is always
scoped to the selected tenant, so the RAG context contains only that tenant's
documents.
"""

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder


def build_context(hits) -> str:
    """Build the context that would be passed to an LLM."""
    return "\n\n".join(hit.text for hit in hits)


def main() -> None:
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    cfg = DynavecConfig(
        vector_bucket="dynavec-demo",
        index="multi-tenant-rag",
        table="dynavec_multi_tenant_rag",
        dimension=embedder.dimension,
        distance_metric="cosine",
        region="us-east-1",
        auto_provision=True,
    )

    db = Dynavec(cfg, embedder=embedder)

    # Each namespace represents one tenant.
    acme = db.namespace("acme")
    globex = db.namespace("globex")

    acme.upsert(
        [
            Document(
                id="refund-policy",
                text="Acme Corp allows customers to request a refund within 30 days.",
            ),
            Document(
                id="support",
                text="Acme Corp customer support is available Monday through Friday.",
            ),
        ]
    )

    globex.upsert(
        [
            Document(
                id="refund-policy",
                text="Globex Inc allows customers to request a refund within 14 days.",
            ),
            Document(
                id="support",
                text="Globex Inc customer support is available 24 hours a day.",
            ),
        ]
    )

    query = "What is the refund period?"

    for tenant_name, tenant in [("acme", acme), ("globex", globex)]:
        hits = tenant.search(query, top_k=2)
        context = build_context(hits)

        print(f"\n--- {tenant_name} ---")
        print("Retrieved context:")
        print(context)

        # Pass `context` and `query` to your preferred LLM to generate the
        # final answer. The important part is that the context is tenant-scoped.
        print("\nRAG prompt context prepared for this tenant.")


if __name__ == "__main__":
    main()
