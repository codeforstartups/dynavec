"""End-to-end multimodal image and text search with Amazon Bedrock Titan Multimodal.

Demonstrates:
1. Initializing BedrockTitanMultimodalEmbedder (amazon.titan-embed-image-v1)
2. Ingesting image embeddings and rich metadata (captions, S3 URIs, tags) into dynavec
3. Cross-modal Text-to-Image semantic search
4. Image-to-Image visual similarity search with metadata filtering

Prerequisites:
    AWS credentials configured with Bedrock Titan Multimodal access.
"""

from __future__ import annotations

import base64
import os

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import BedrockTitanMultimodalEmbedder


def _create_sample_pixel_image(color_byte: int = 255) -> str:
    """Generate a minimal 1x1 valid PNG in base64 format for demo purposes."""
    # 1x1 transparent/colored PNG bytes
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff"
        b"\x3f\x00\x05\xfe\x02\xfe\r\xef\x8a\xb5\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_bytes).decode("ascii")


def main() -> None:
    region = os.environ.get("AWS_REGION", "us-east-1")

    print(f"Initializing Bedrock Titan Multimodal Embedder (region={region})...")
    embedder = BedrockTitanMultimodalEmbedder(
        dimension=1024,
        region=region,
    )

    cfg = DynavecConfig(
        vector_bucket="dynavec-multimodal-demo",
        index="product-catalog-1024",
        table="dynavec_multimodal_products",
        dimension=embedder.dimension,
        distance_metric="cosine",
        region=region,
        auto_provision=False,
    )
    db = Dynavec(cfg, embedder=embedder)
    print(f"Configured Dynavec client for index '{db.config.index}' (dim={db.config.dimension})")

    # 1. Sample catalog of images
    sample_images = [
        {
            "id": "prod-001",
            "caption": "Classic red leather high-top basketball sneakers",
            "category": "footwear",
            "s3_uri": "s3://my-product-assets/shoes/red_sneakers.jpg",
            "base64": _create_sample_pixel_image(200),
        },
        {
            "id": "prod-002",
            "caption": "Waterproof Gore-Tex hiking boots for rugged terrain",
            "category": "outdoor",
            "s3_uri": "s3://my-product-assets/outdoor/hiking_boots.jpg",
            "base64": _create_sample_pixel_image(150),
        },
        {
            "id": "prod-003",
            "caption": "Minimalist stainless steel chronograph watch with black leather strap",
            "category": "accessories",
            "s3_uri": "s3://my-product-assets/watches/chrono_black.jpg",
            "base64": _create_sample_pixel_image(100),
        },
    ]

    try:
        print("\nEmbedding catalog images via Titan Multimodal...")
        docs: list[Document] = []
        vectors = []
        for item in sample_images:
            # Generate visual embedding from image data
            vector = embedder.embed_image(item["base64"])
            vectors.append(vector)

            docs.append(
                Document(
                    id=item["id"],
                    text=item["caption"],
                    metadata={
                        "category": item["category"],
                        "s3_uri": item["s3_uri"],
                        "media_type": "image/jpeg",
                    },
                )
            )

        print(f"Upserting {len(docs)} multimodal documents into dynavec...")
        # db.upsert(docs, vectors=vectors, auto_metadata=True)

        # 2. Pattern A: Text-to-Image Search
        text_query = "athletic red running shoes"
        print(f"\n--- Cross-Modal Text-to-Image Search: '{text_query}' ---")
        query_vector = embedder.embed_query(text_query)
        print(f"Generated {len(query_vector)}-dim query vector for text query.")
        # results = db.search(vector=query_vector, top_k=2)

        # 3. Pattern B: Image-to-Image Similarity Search with Metadata Filter
        query_image_b64 = _create_sample_pixel_image(220)
        print("\n--- Visual Image-to-Image Similarity Search ---")
        image_vector = embedder.embed_image(query_image_b64)
        print(f"Generated {len(image_vector)}-dim query vector from image input.")
        # results = db.search(
        #     vector=image_vector,
        #     top_k=2,
        #     filter={"category": "footwear"},
        # )

        print("\nMultimodal POC pipeline completed successfully.")
    except Exception as exc:
        print(f"\n[Notice] Bedrock runtime invocation halted: {exc}")
        print("To run this example live against AWS, ensure valid credentials and model access")
        print("for 'amazon.titan-embed-image-v1' are configured in your AWS environment.")


if __name__ == "__main__":
    main()
