"""Ingest a PDF into dynavec (requires AWS credentials + S3 Vectors).

    pip install "dynavec[ingest,sentence-transformers]"
    python examples/pdf_ingestion.py path/to/document.pdf
"""

import sys

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.ingest import PDFSource, ingest

if len(sys.argv) != 2:
    raise SystemExit(
        "usage: python examples/pdf_ingestion.py path/to/document.pdf"
    )

# Local embedding model — no external API key required.
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="pdf-ingestion",
    table="dynavec_pdf_ingestion",
    dimension=embedder.dimension,
    distance_metric="cosine",
    region="us-east-1",
    auto_provision=True,
)

db = Dynavec(cfg, embedder=embedder)

source = PDFSource(sys.argv[1])

count = ingest(db, source, namespace="pdf-demo")

print(f"Ingested {count} chunks")
