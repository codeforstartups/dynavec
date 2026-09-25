"""Ingest documents from an Amazon S3 bucket into existing AWS resources.

    pip install "dynavec[ingest,openai]"
    python examples/ingest_s3.py my-source-bucket --prefix documents/ --preview
    python examples/ingest_s3.py my-source-bucket --prefix docs/ --bucket my-vectors --index docs --table docs

Preview mode reads and parses objects from the source S3 bucket and prints extracted
record IDs, text length, and metadata without writing to S3 Vectors or DynamoDB.
"""

import argparse

from dynavec import Dynavec, DynavecConfig
from dynavec.embeddings import OpenAIEmbedder
from dynavec.ingest import S3Source, ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_bucket", help="S3 bucket holding documents to ingest")
    parser.add_argument("--prefix", default="", help="Optional S3 object prefix filter")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional file extension filter (e.g. .md, .pdf, .txt)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print parsed records without embedding or writing",
    )
    parser.add_argument("--bucket", help="Target S3 vector bucket for Dynavec")
    parser.add_argument("--index", help="Target vector index name")
    parser.add_argument("--table", help="Target DynamoDB table name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--namespace", default="docs", help="Target namespace")
    args = parser.parse_args()

    source = S3Source(
        args.source_bucket,
        prefix=args.prefix,
        suffix=args.suffix,
    )

    if args.preview:
        print(f"Listing and parsing objects from s3://{args.source_bucket}/{args.prefix}...")
        count = 0
        for record in source:
            count += 1
            sample = record.text[:80].replace("\n", " ")
            print(f"[{count}] {record.id} ({len(record.text)} chars): {sample}...")
            print(f"    Metadata: {record.metadata}")
        print(f"\nDiscovered {count} records ready for ingestion.")
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
