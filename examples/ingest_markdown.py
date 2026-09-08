"""Ingest a Markdown/text folder into existing AWS resources.

    pip install "dynavec[ingest,openai]"
    python examples/ingest_markdown.py ./notes --preview
    python examples/ingest_markdown.py ./notes --bucket my-vectors --index docs --table docs

Writes require AWS credentials and OPENAI_API_KEY. The index must use 1536
dimensions for text-embedding-3-small. Preview only reads local files.
"""

import argparse

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import OpenAIEmbedder
from dynavec.ingest import MarkdownSource, ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--glob", default="**/*")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--bucket")
    parser.add_argument("--index")
    parser.add_argument("--table")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--namespace", default="docs")
    args = parser.parse_args()
    source = MarkdownSource(args.directory, glob=args.glob)

    if args.preview:
        for record in source:
            print(f"{record.id}: {len(record.text)} characters; {record.metadata}")
        return
    if not all((args.bucket, args.index, args.table)):
        parser.error("--bucket, --index and --table are required unless --preview is used")

    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    config = DynavecConfig(
        vector_bucket=args.bucket,
        index=args.index,
        table=args.table,
        dimension=embedder.dimension,
        region=args.region,
    )
    with Dynavec(config, embedder=embedder) as db:
        count = ingest(db, source, namespace=args.namespace)
    print(f"Upserted {count} chunks into namespace {args.namespace!r}")


if __name__ == "__main__":
    main()
