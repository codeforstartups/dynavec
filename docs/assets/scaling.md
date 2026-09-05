# dynavec scaling: dimensions × vector count (100K → 1B)

_Cost from the cost model (APPROX public list prices). Recall/latency unaffected by this sweep._

### 384-dim embeddings — $/month (1,000,000 queries/mo)

| Product | 100K | 1M | 10M | 100M | 1B |
|---|---|---|---|---|---|
| dynavec | **$3** | **$3** | **$5** | **$24** | **$212** |
| Pinecone | $8 | $9 | $13 | $56 | $481 |
| OpenSearch | $701 | $701 | $701 | $2,106 | $21,058 |
| Qdrant | $160 | $160 | $320 | $2,240 | $21,600 |
| Weaviate | $175 | $175 | $350 | $2,450 | $23,625 |
| Milvus/Zilliz | $150 | $150 | $300 | $2,100 | $20,250 |
| _raw float32 size_ | 0 GB | 1 GB | 14 GB | 143 GB | 1,431 GB |

### 768-dim embeddings — $/month (1,000,000 queries/mo)

| Product | 100K | 1M | 10M | 100M | 1B |
|---|---|---|---|---|---|
| dynavec | **$3** | **$3** | **$6** | **$32** | **$298** |
| Pinecone | $9 | $9 | $18 | $103 | $953 |
| OpenSearch | $701 | $701 | $701 | $4,212 | $41,941 |
| Qdrant | $160 | $160 | $480 | $4,320 | $43,040 |
| Weaviate | $175 | $175 | $525 | $4,725 | $47,075 |
| Milvus/Zilliz | $150 | $150 | $450 | $4,050 | $40,350 |
| _raw float32 size_ | 0 GB | 3 GB | 29 GB | 286 GB | 2,861 GB |

### 1024-dim embeddings — $/month (1,000,000 queries/mo)

| Product | 100K | 1M | 10M | 100M | 1B |
|---|---|---|---|---|---|
| dynavec | **$3** | **$3** | **$6** | **$38** | **$355** |
| Pinecone | $9 | $10 | $21 | $134 | $1,267 |
| OpenSearch | $701 | $701 | $702 | $5,616 | $55,805 |
| Qdrant | $160 | $160 | $640 | $5,760 | $57,280 |
| Weaviate | $175 | $175 | $700 | $6,300 | $62,650 |
| Milvus/Zilliz | $150 | $150 | $600 | $5,400 | $53,700 |
| _raw float32 size_ | 0 GB | 4 GB | 38 GB | 381 GB | 3,815 GB |

### 1536-dim embeddings — $/month (1,000,000 queries/mo)

| Product | 100K | 1M | 10M | 100M | 1B |
|---|---|---|---|---|---|
| dynavec | **$3** | **$3** | **$8** | **$50** | **$469** |
| Pinecone | $9 | $10 | $27 | $197 | $1,897 |
| OpenSearch | $701 | $701 | $877 | $8,423 | $83,708 |
| Qdrant | $160 | $160 | $960 | $8,640 | $85,920 |
| Weaviate | $175 | $175 | $1,050 | $9,450 | $93,975 |
| Milvus/Zilliz | $150 | $150 | $900 | $8,100 | $80,550 |
| _raw float32 size_ | 1 GB | 6 GB | 57 GB | 572 GB | 5,722 GB |

### 3072-dim embeddings — $/month (1,000,000 queries/mo)

| Product | 100K | 1M | 10M | 100M | 1B |
|---|---|---|---|---|---|
| dynavec | **$3** | **$4** | **$11** | **$84** | **$813** |
| Pinecone | $9 | $12 | $46 | $386 | $3,785 |
| OpenSearch | $701 | $701 | $1,755 | $16,847 | $167,415 |
| Qdrant | $160 | $320 | $1,760 | $17,280 | $171,680 |
| Weaviate | $175 | $350 | $1,925 | $18,900 | $187,775 |
| Milvus/Zilliz | $150 | $300 | $1,650 | $16,200 | $160,950 |
| _raw float32 size_ | 1 GB | 11 GB | 114 GB | 1,144 GB | 11,444 GB |
