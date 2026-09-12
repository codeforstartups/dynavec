"""Tests for metadata splitting / auto-generation / filter building."""

import pytest

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


@pytest.mark.parametrize(
    ("user_filter", "namespace", "expected"),
    [
        # None and empty filter
        (None, "default", {NS_METADATA_KEY: "default"}),
        ({}, "production", {NS_METADATA_KEY: "production"}),
        # Bare equality
        (
            {"genre": "scifi"},
            "ns1",
            {"$and": [{"genre": "scifi"}, {NS_METADATA_KEY: "ns1"}]},
        ),
        # $eq operator
        (
            {"status": {"$eq": "active"}},
            "ns_app",
            {"$and": [{"status": {"$eq": "active"}}, {NS_METADATA_KEY: "ns_app"}]},
        ),
        # $ne operator
        (
            {"archived": {"$ne": True}},
            "ns_v1",
            {"$and": [{"archived": {"$ne": True}}, {NS_METADATA_KEY: "ns_v1"}]},
        ),
        # Numeric comparison operators ($gte, $gt, $lte, $lt)
        (
            {"rating": {"$gte": 4.5}},
            "catalog",
            {"$and": [{"rating": {"$gte": 4.5}}, {NS_METADATA_KEY: "catalog"}]},
        ),
        (
            {"price": {"$lt": 50.0}},
            "catalog",
            {"$and": [{"price": {"$lt": 50.0}}, {NS_METADATA_KEY: "catalog"}]},
        ),
        (
            {"score": {"$gt": 0.8}, "views": {"$lte": 1000}},
            "analytics",
            {
                "$and": [
                    {"score": {"$gt": 0.8}, "views": {"$lte": 1000}},
                    {NS_METADATA_KEY: "analytics"},
                ]
            },
        ),
        # $in and $nin operators
        (
            {"tag": {"$in": ["python", "aws", "vector"]}},
            "dev",
            {
                "$and": [
                    {"tag": {"$in": ["python", "aws", "vector"]}},
                    {NS_METADATA_KEY: "dev"},
                ]
            },
        ),
        (
            {"role": {"$nin": ["guest", "banned"]}},
            "users",
            {
                "$and": [
                    {"role": {"$nin": ["guest", "banned"]}},
                    {NS_METADATA_KEY: "users"},
                ]
            },
        ),
        # $or operator combinations
        (
            {"$or": [{"category": "news"}, {"category": "tech"}]},
            "media",
            {
                "$and": [
                    {"$or": [{"category": "news"}, {"category": "tech"}]},
                    {NS_METADATA_KEY: "media"},
                ]
            },
        ),
        (
            {"$or": [{"price": {"$gte": 100}}, {"rating": {"$gte": 4.8}}]},
            "products",
            {
                "$and": [
                    {"$or": [{"price": {"$gte": 100}}, {"rating": {"$gte": 4.8}}]},
                    {NS_METADATA_KEY: "products"},
                ]
            },
        ),
        (
            {
                "$or": [
                    {"category": {"$in": ["electronics", "computers"]}},
                    {"discount": {"$gte": 0.2}},
                ]
            },
            "store",
            {
                "$and": [
                    {
                        "$or": [
                            {"category": {"$in": ["electronics", "computers"]}},
                            {"discount": {"$gte": 0.2}},
                        ]
                    },
                    {NS_METADATA_KEY: "store"},
                ]
            },
        ),
        # Top-level $and list extension (should not double wrap)
        (
            {"$and": [{"a": 1}]},
            "ns1",
            {"$and": [{"a": 1}, {NS_METADATA_KEY: "ns1"}]},
        ),
        (
            {
                "$and": [
                    {"rating": {"$gte": 4.0}},
                    {"tag": {"$in": ["featured", "trending"]}},
                ]
            },
            "catalog",
            {
                "$and": [
                    {"rating": {"$gte": 4.0}},
                    {"tag": {"$in": ["featured", "trending"]}},
                    {NS_METADATA_KEY: "catalog"},
                ]
            },
        ),
        # Complex nested $and + $or + $in + $gte combinations
        (
            {
                "$and": [
                    {"$or": [{"tier": "premium"}, {"tier": "vip"}]},
                    {"spend": {"$gte": 500}},
                    {"country": {"$in": ["US", "CA", "UK"]}},
                ]
            },
            "customers",
            {
                "$and": [
                    {"$or": [{"tier": "premium"}, {"tier": "vip"}]},
                    {"spend": {"$gte": 500}},
                    {"country": {"$in": ["US", "CA", "UK"]}},
                    {NS_METADATA_KEY: "customers"},
                ]
            },
        ),
    ],
)
def test_build_s3_filter_operator_combinations(user_filter, namespace, expected):
    """Table-driven tests covering $and, $or, $in, $gte and other operators."""
    assert build_s3_filter(user_filter, namespace) == expected

