"""Tests for Dynavec.describe() — index + table config snapshot."""

from unittest.mock import MagicMock

import boto3
from botocore.stub import Stubber

from dynavec.client import Dynavec
from dynavec.config import DynavecConfig
from dynavec.models import IndexInfo
from dynavec.stores.s3vectors import S3VectorsStore


def _config(**kwargs):
    defaults = dict(
        vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
    )
    defaults.update(kwargs)
    return DynavecConfig(**defaults)


def _index_payload(**overrides):
    payload = {
        "vectorBucketName": "test-bucket",
        "indexName": "test-index",
        "indexArn": "arn:aws:s3vectors:us-east-1:123456789012:bucket/test-bucket/index/test-index",
        "creationTime": 0,
        "dataType": "float32",
        "dimension": 4,
        "distanceMetric": "cosine",
    }
    payload.update(overrides)
    return payload


def _table_payload(**overrides):
    payload = {
        "TableName": "test-table",
        "TableStatus": "ACTIVE",
        "ItemCount": 42,
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------- store: get_index()
def test_get_index_calls_s3vectors_with_bucket_and_index_name():
    store = S3VectorsStore.__new__(S3VectorsStore)
    store._config = _config()
    store._client = MagicMock()
    store._client.get_index.return_value = {"index": _index_payload()}

    result = store.get_index()

    store._client.get_index.assert_called_once_with(
        vectorBucketName="test-bucket", indexName="test-index"
    )
    assert result["index"]["dimension"] == 4


# ------------------------------------------------------- client: describe()
def _client_with_fakes(index_payload, table_payload):
    db = Dynavec.__new__(Dynavec)
    db.config = _config()

    db._vectors = MagicMock()
    db._vectors.get_index.return_value = {"index": index_payload}

    fake_ddb_client = MagicMock()
    fake_ddb_client.describe_table.return_value = {"Table": table_payload}
    db._docs = MagicMock()
    db._docs._ddb.meta.client = fake_ddb_client

    return db


def test_describe_returns_populated_index_info():
    db = _client_with_fakes(_index_payload(), _table_payload())

    info = db.describe()

    assert isinstance(info, IndexInfo)
    assert info.vector_bucket == "test-bucket"
    assert info.index == "test-index"
    assert info.dimension == 4
    assert info.distance_metric == "cosine"
    assert info.table == "test-table"
    assert info.table_status == "ACTIVE"
    assert info.item_count == 42
    assert info.non_filterable_keys == []


def test_describe_handles_missing_metadata_configuration():
    # no metadataConfiguration key at all -> should default cleanly, not KeyError
    db = _client_with_fakes(_index_payload(), _table_payload())

    info = db.describe()

    assert info.non_filterable_keys == []


def test_describe_surfaces_non_filterable_keys_when_present():
    index_payload = _index_payload(
        metadataConfiguration={"nonFilterableMetadataKeys": ["_dv_text"]}
    )
    db = _client_with_fakes(index_payload, _table_payload())

    info = db.describe()

    assert info.non_filterable_keys == ["_dv_text"]


# --------------------------------------------------- real service model check
def test_get_index_request_matches_s3vectors_service_model():
    """Stubber validates our params against the actual botocore s3vectors model."""
    client = boto3.client("s3vectors", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_response(
        "get_index",
        {"index": _index_payload()},
        {"vectorBucketName": "test-bucket", "indexName": "test-index"},
    )
    stubber.activate()

    store = S3VectorsStore.__new__(S3VectorsStore)
    store._config = _config()
    store._client = client

    result = store.get_index()
    stubber.assert_no_pending_responses()
    assert result["index"]["indexName"] == "test-index"