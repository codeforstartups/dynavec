"""Use dynavec as a Semantic Kernel VectorStore.

pip install "dynavec[sentence-transformers,semantic-kernel]"
python examples/semantic_kernel_vector_store.py
"""

import asyncio

from semantic_kernel.data.vector import (
    FieldTypes,
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.semantic_kernel import DynavecStore


async def main():
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    cfg = DynavecConfig(
        vector_bucket="dynavec-demo",
        index="sk-vector-store",
        table="dynavec_sk_vector_store",
        dimension=embedder.dimension,
        region="us-east-1",
        auto_provision=True,
    )

    db = Dynavec(cfg, embedder=embedder)
    store = DynavecStore(db)

    definition = VectorStoreCollectionDefinition(
        fields=[
            VectorStoreField(field_type=FieldTypes.KEY, name="id"),
            VectorStoreField(field_type=FieldTypes.DATA, name="text"),
            VectorStoreField(field_type=FieldTypes.DATA, name="category"),
            VectorStoreField(
                field_type=FieldTypes.VECTOR,
                name="embedding",
                dimensions=embedder.dimension,
            ),
        ],
    )

    collection = store.get_collection(
        record_type=dict,
        definition=definition,
        collection_name="kb",
    )

    texts = [
        "Semantic Kernel is a framework for building AI applications.",
        "Retrieval-augmented generation grounds answers in your documents.",
    ]
    vectors = embedder.embed_documents(texts)

    await collection.upsert(
        [
            {
                "id": "sk-1",
                "text": texts[0],
                "category": "ai",
                "embedding": vectors[0],
            },
            {
                "id": "sk-2",
                "text": texts[1],
                "category": "rag",
                "embedding": vectors[1],
            },
        ]
    )

    query_vector = embedder.embed_query("How can I build an AI application?")

    results = await collection.search(
        vector=query_vector,
        top=2,
    )

    async for result in results.results:
        print(result.score, "-", result.record["text"])


if __name__ == "__main__":
    asyncio.run(main())
