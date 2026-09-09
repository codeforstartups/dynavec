"""Tests for DynavecConfig.max_pool_connections plumbing (issue #54)."""

from unittest.mock import MagicMock

import pytest

from dynavec.cache import DynamoDBCache
from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore
from dynavec.provisioning import ensure_index, ensure_table, ensure_vector_bucket
from dynavec.stores.dynamodb import DynamoDBStore
from dynavec.stores.s3vectors import S3VectorsStore


def _config(**kwargs):
    defaults = dict(vector_bucket="b", index="i", table="t", dimension=4)
    defaults.update(kwargs)
    return DynavecConfig(**defaults)


def test_max_pool_connections_defaults_to_none():
    assert _config().max_pool_connections is None
    assert _config().botocore_config() is None


def test_max_pool_connections_validation():
    with pytest.raises(ValueError):
        _config(max_pool_connections=0)
    with pytest.raises(ValueError):
        _config(max_pool_connections=-5)


def test_botocore_config_carries_pool_size():
    cfg = _config(max_pool_connections=50)
    client_config = cfg.botocore_config()
    assert client_config is not None
    assert client_config.max_pool_connections == 50


def _mock_session():
    session = MagicMock()
    resource = MagicMock()
    resource.Table.return_value = MagicMock()
    session.resource.return_value = resource
    session.client.return_value = MagicMock()
    return session


def test_dynamodb_store_passes_config():
    session = _mock_session()
    DynamoDBStore(_config(max_pool_connections=25), boto_session=session)
    _, kwargs = session.resource.call_args
    assert kwargs["config"].max_pool_connections == 25


def test_dynamodb_store_omits_config_by_default():
    session = _mock_session()
    DynamoDBStore(_config(), boto_session=session)
    _, kwargs = session.resource.call_args
    assert "config" not in kwargs


def test_s3vectors_store_passes_config():
    session = _mock_session()
    S3VectorsStore(_config(max_pool_connections=25), boto_session=session)
    _, kwargs = session.client.call_args
    assert kwargs["config"].max_pool_connections == 25


def test_s3vectors_store_omits_config_by_default():
    session = _mock_session()
    S3VectorsStore(_config(), boto_session=session)
    _, kwargs = session.client.call_args
    assert "config" not in kwargs


def test_graph_store_passes_config():
    session = _mock_session()
    GraphStore(_config(max_pool_connections=12), boto_session=session)
    _, kwargs = session.resource.call_args
    assert kwargs["config"].max_pool_connections == 12


def test_dynamodb_cache_passes_config():
    session = _mock_session()
    DynamoDBCache(_config(max_pool_connections=12), boto_session=session)
    _, kwargs = session.resource.call_args
    assert kwargs["config"].max_pool_connections == 12


def test_provisioning_passes_config():
    session = _mock_session()
    cfg = _config(max_pool_connections=30)
    ensure_vector_bucket(cfg, boto_session=session)
    ensure_index(cfg, boto_session=session)
    ensure_table(cfg, boto_session=session)
    for call in session.client.call_args_list:
        _, kwargs = call
        assert kwargs["config"].max_pool_connections == 30
