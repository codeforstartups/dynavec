"""Ingest a DOCX file into dynavec (requires AWS credentials + S3 Vectors).

    pip install "dynavec[ingest,sentence-transformers]"
    python examples/docx_ingestion.py path/to/document.docx
"""

import sys

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.ingest import DocxSource, ingest

if len(sys.argv) != 2:
    raise SystemExit(
        "usage: python examples/docx_ingestion.py path/to/document.docx"
    )

# Local embedding model — no external API key required.
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="docx-ingestion",
    table="dynavec_docx_ingestion",
    dimension=embedder.dimension,
    distance_metric="cosine",
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

source = DocxSource(sys.argv[1])

count = ingest(db, source, namespace="docx-demo")

print(f"Ingested {count} chunks")
