# Multimodal & Image Embeddings in dynavec

## 1. Overview & Problem Statement

Modern vector databases and retrieval-augmented generation (RAG) pipelines have predominantly focused on single-modal text search. However, real-world enterprise applications—such as e-commerce visual search, brand asset management, medical imaging, and multi-modal document understanding—require indexing and querying non-text modalities alongside rich structured metadata.

Amazon Bedrock provides the **Titan Multimodal Embeddings** foundation model (`amazon.titan-embed-image-v1`). This model maps text queries, images (JPEG, PNG, GIF, WebP), or combined text-and-image inputs into a **single, shared vector space**.

Because `dynavec` decouples low-cost scalable vector similarity search (via Amazon S3 Vectors) from low-latency metadata filtering and document retrieval (via Amazon DynamoDB), it is uniquely suited for serverless, in-account multimodal search.

This document outlines the architecture, data structures, ingestion protocols, and cross-modal query workflows for multimodal embeddings in `dynavec`.

---

## 2. System Architecture

```
                                  +---------------------------------------+
                                  |            User Application           |
                                  +---------------------------------------+
                                           |                      |
                    Text Query ("red sports car")     Image Query (photo bytes)
                                           |                      |
                                           v                      v
                                  +---------------------------------------+
                                  |     BedrockTitanMultimodalEmbedder    |
                                  |      (amazon.titan-embed-image-v1)    |
                                  +---------------------------------------+
                                                      |
                                           Embedding Vector (1024-d)
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |             Dynavec Client            |
                                  +---------------------------------------+
                                           |                      |
                       Vector ANN Search   |                      | Metadata Read/Write
                                           v                      v
                             +------------------------+  +------------------------+
                             |   Amazon S3 Vectors    |  |    Amazon DynamoDB     |
                             | - 1024/384/256-d index |  | - Image URLs / S3 URIs |
                             | - Cosine similarity    |  | - Captions & tags      |
                             | - Partition namespaces |  | - EXIF & dimensions    |
                             +------------------------+  +------------------------+
                                                                  |
                                                                  v
                                                         +------------------------+
                                                         | Amazon S3 Asset Bucket |
                                                         | (Raw image binary data)|
                                                         +------------------------+
```

### Core Architecture Principles

1. **Zero Raw Binary in Vector / Document Stores**:
   Raw image files (megabytes in size) remain in standard Amazon S3 buckets (`s3://my-assets-bucket/...`).
2. **Dense Vector Embeddings in S3 Vectors**:
   Bedrock Titan Multimodal converts the image into a compact dense embedding (e.g. 1024 float32 dimensions = 4 KB), which is indexed directly in Amazon S3 Vectors.
3. **Structured Document Metadata in DynamoDB**:
   Image resolution, file format, captions, tags, user IDs, and direct S3 URIs are stored in DynamoDB for sub-10ms lookup and boolean filtering (`filter={"category": "automotive"}`).

---

## 3. Amazon Titan Multimodal Model Details

The `amazon.titan-embed-image-v1` foundation model supports:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Model ID** | `amazon.titan-embed-image-v1` | Available in `us-east-1`, `us-west-2`, `ap-northeast-1`, etc. |
| **Output Dimensions** | `256`, `384`, `1024` (Default) | Configurable via `embeddingConfig.outputEmbeddingLength` |
| **Supported Image Formats** | JPEG, PNG, GIF, WebP | Max 2048x2048 resolution, max 5 MB file size |
| **Supported Inputs** | Text only, Image only, or Text + Image | All map to the same joint metric space |
| **Distance Metric** | Cosine / Dot Product | Embeddings are normalized by default |

### Request Payload Structure:
```json
{
  "inputText": "Optional descriptive caption or query text",
  "inputImage": "Optional base64-encoded image string",
  "embeddingConfig": {
    "outputEmbeddingLength": 1024
  }
}
```

---

## 4. Ingestion & Storage Workflow

```
[Raw Image File / URL]
         |
         v
1. Encode image bytes to Base64
         |
         v
2. Call BedrockTitanMultimodalEmbedder.embed_image(...)
         |
         v (1024-dim Vector)
3. Construct Document:
   - id: "img-001"
   - text: "Red Italian sports car parked on coastal road"
   - metadata: {
       "s3_uri": "s3://assets/cars/ferrari.jpg",
       "format": "jpeg",
       "width": 1920,
       "height": 1080,
       "tags": ["automotive", "sports-car"]
     }
         |
         v
4. db.upsert([doc], vectors=[vector])
   ├──> Vector (1024-d) ──────> Stored in S3 Vectors index
   └──> Metadata & Text ──────> Stored in DynamoDB table
```

---

## 5. Cross-Modal Query Execution

Because the vector representations of text and images share the same embedding geometry, `dynavec` supports two primary cross-modal query patterns:

### Pattern A: Text-to-Image Search (Semantic Concept Retrieval)
The user enters natural language ("sunset over snowy mountain peaks").
1. Embedder calls Titan Multimodal with `inputText="sunset over snowy mountain peaks"`.
2. S3 Vectors retrieves nearest neighbor vector IDs corresponding to indexed images.
3. DynamoDB fetches image metadata (captions, S3 links) for top-$k$ matches.

### Pattern B: Image-to-Image Search (Visual Similarity)
The user uploads a query image (or a bounding box crop).
1. Embedder calls Titan Multimodal with `inputImage=<base64_encoded_query_image>`.
2. S3 Vectors searches for visually and semantically similar images.
3. DynamoDB retrieves candidate matches with optional category filtering.

---

## 6. Dimension Selection & Cost Analysis

### Dimension Tradeoffs

* **1024 dimensions (Default)**:
  * Highest retrieval accuracy and fine-grained visual concept distinction.
  * Vector storage: ~4 KB per image.
* **384 dimensions**:
  * Balances memory footprint and S3 Vectors scan throughput.
  * Matches popular small text embedding models (`all-MiniLM-L6-v2`).
* **256 dimensions**:
  * Ultra-low storage footprint (~1 KB per image), ideal for billion-scale visual indexing.

### Cost Profile (AWS Native)
* **Titan Multimodal Ingestion**: ~$0.00006 per image (approx. $60 per 1 million images).
* **S3 Vectors Storage**: ~$0.023 / GB / month. 1M 1024-dim vectors occupy ~4 GB ($0.10 / month).
* **DynamoDB Metadata**: Serverless on-demand or provisioned capacity with zero server maintenance.

---

## 7. API Design in `dynavec.embeddings`

The `BedrockTitanMultimodalEmbedder` implements the `Embedder` protocol and provides specialized image embedding helpers:

```python
from dynavec.embeddings import BedrockTitanMultimodalEmbedder

embedder = BedrockTitanMultimodalEmbedder(
    region="us-east-1",
    dimension=1024,
)

# Text embedding (for text-to-image queries)
query_vec = embedder.embed_query("vintage sports car")

# Image embedding (from file path, bytes, or base64)
image_vec = embedder.embed_image("path/to/image.jpg")

# Batch image embedding
batch_vecs = embedder.embed_images(["img1.jpg", "img2.png"])

# Combined multimodal embedding
joint_vec = embedder.embed_multimodal(
    text="car with blue custom paint",
    image="path/to/car.jpg",
)
```
