"""End-to-end integration test (opt-in) — runs against real AWS or a local emulator.

Auto-provisions an S3 vector bucket + index + DynamoDB table, upserts vectors,
queries, verifies metadata filtering + document hydration, then deletes every
resource it created.

Against real AWS (costs money, uses YOUR account):
    export DYNAVEC_LIVE=1
    export AWS_REGION=us-east-1          # a region where S3 Vectors is available
    pytest tests/integration/test_live_aws.py -v -s

Against a local emulator (free, no account) — see "Local development" in the README:
    docker compose up -d --wait
    export AWS_ENDPOINT_URL=http://localhost:4566
    pytest tests/integration/test_live_aws.py -v -s

boto3 honours AWS_ENDPOINT_URL natively, so dynavec needs no endpoint plumbing.
Skipped by default: the normal suite needs no credentials, no Docker, no cost.
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest

LOCAL_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")

if LOCAL_ENDPOINT:
    # Emulators accept any non-empty credentials, but boto3 still refuses to sign
    # without them. setdefault so a real profile/role is never overridden.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

pytestmark = pytest.mark.skipif(
    os.environ.get("DYNAVEC_LIVE") != "1" and not LOCAL_ENDPOINT,
    reason="set DYNAVEC_LIVE=1 (real AWS) or AWS_ENDPOINT_URL (local emulator) to run",
)

REGION = os.environ.get("AWS_REGION", "us-east-1")
DIM = 32
N = 40
NS = "it"  # searches must target the same namespace the upserts wrote to


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def live_db():
    import boto3

    from dynavec import Dynavec, DynavecConfig

    sfx = uuid.uuid4().hex[:8]
    cfg = DynavecConfig(
        vector_bucket=f"dynavec-it-{sfx}",
        index=f"it{sfx}",
        table=f"dynavec_it_{sfx}",
        dimension=DIM,
        region=REGION,
        auto_provision=True,
    )
    db = Dynavec(cfg)
    yield db
    # ---- teardown: remove everything we created ----
    s3v = boto3.client("s3vectors", region_name=REGION)
    ddb = boto3.client("dynamodb", region_name=REGION)
    for fn in (
        lambda: s3v.delete_index(vectorBucketName=cfg.vector_bucket, indexName=cfg.index),
        lambda: s3v.delete_vector_bucket(vectorBucketName=cfg.vector_bucket),
        lambda: ddb.delete_table(TableName=cfg.table),
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"teardown warning: {e}")
    db.close()


def _query_with_retry(db, vector, top_k, filter=None, namespace=NS, tries=8, delay=3):
    if LOCAL_ENDPOINT:
        delay = 0  # emulators are immediately consistent; don't burn 24s in CI
    for _ in range(tries):
        hits = db.search(vector=vector, top_k=top_k, filter=filter, namespace=namespace)
        if hits:
            return hits
        time.sleep(delay)  # S3 Vectors is eventually consistent after ingest
    return []


def test_provision_upsert_search_roundtrip(live_db):
    from dynavec import Document

    rng = np.random.default_rng(0)
    vecs = _unit(rng, N, DIM)
    docs = [
        Document(
            id=f"v{i}",
            vector=vecs[i].tolist(),
            text=f"document number {i}",
            metadata={"even": (i % 2 == 0), "idx": i},
        )
        for i in range(N)
    ]
    live_db.upsert(docs, namespace=NS)

    # nearest neighbor of vec[0] should be v0 itself
    hits = _query_with_retry(live_db, vecs[0].tolist(), top_k=5)
    assert hits, "no results returned from live S3 Vectors index"
    assert hits[0].id == "v0"
    assert hits[0].text == "document number 0"   # hydrated from DynamoDB

    # metadata pre-filter must scope results
    even_hits = _query_with_retry(live_db, vecs[0].tolist(), top_k=10, filter={"even": True})
    assert even_hits
    assert all(h.metadata.get("even") is True for h in even_hits)


def test_update_and_delete(live_db):
    from dynavec import Document

    rng = np.random.default_rng(1)
    v = _unit(rng, 1, DIM)[0].tolist()
    live_db.upsert([Document(id="u1", vector=v, text="original", metadata={"tag": "a"})], namespace=NS)

    live_db.update("u1", namespace=NS, metadata={"tag": "b"}, merge_metadata=False)
    got = live_db.get(["u1"], namespace=NS)[0]
    assert got.metadata["tag"] == "b"

    live_db.delete(["u1"], namespace=NS)
    assert live_db.get(["u1"], namespace=NS) == []
