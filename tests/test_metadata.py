"""Tests for metadata splitting / auto-generation / filter building."""

from dynavec.config import NS_METADATA_KEY, TEXT_METADATA_KEY, DynavecConfig
from dynavec.metadata import (
    build_s3_filter,
    generate_auto_metadata,
    split_metadata,
)


def _cfg(**kw):
    base = dict(vector_bucket="b", index="i", table="t", dimension=8)
    base.update(kw)
    return DynavecConfig(**base)


def test_auto_metadata_fields():
    meta = generate_auto_metadata("hello world foo")
    assert meta["word_count"] == 3
    assert meta["char_count"] == len("hello world foo")
    assert "content_hash" in meta
    assert "created_at" in meta


def test_split_default_pushes_scalars_and_ns():
    cfg = _cfg()
    s3, ddb = split_metadata({"lang": "en", "score": 0.5, "nested": {"x": 1}}, cfg, "ns1")
    # scalars go to S3 Vectors; nested dict stays only in DynamoDB
    assert s3["lang"] == "en"
    assert s3["score"] == 0.5
    assert "nested" not in s3
    assert s3[NS_METADATA_KEY] == "ns1"
    # DynamoDB keeps everything
    assert ddb["nested"] == {"x": 1}


def test_split_respects_filterable_keys_allowlist():
    cfg = _cfg(filterable_keys=["lang"])
    s3, ddb = split_metadata({"lang": "en", "author": "abhi"}, cfg, "ns")
    assert "lang" in s3
    assert "author" not in s3  # excluded from S3 Vectors, still in DynamoDB
    assert ddb["author"] == "abhi"


def test_split_text_mirror_optional():
    cfg = _cfg(store_text_in_s3vectors=True, text_mirror_max_chars=5)
    s3, _ = split_metadata({}, cfg, "ns", text="abcdefgh")
    assert s3[TEXT_METADATA_KEY] == "abcde"


def test_build_filter_wraps_with_namespace():
    f = build_s3_filter({"genre": "scifi"}, "ns1")
    assert f == {"$and": [{"genre": "scifi"}, {NS_METADATA_KEY: "ns1"}]}


def test_build_filter_none():
    assert build_s3_filter(None, "ns1") == {NS_METADATA_KEY: "ns1"}


def test_build_filter_extends_existing_and():
    f = build_s3_filter({"$and": [{"a": 1}]}, "ns1")
    assert f == {"$and": [{"a": 1}, {NS_METADATA_KEY: "ns1"}]}
