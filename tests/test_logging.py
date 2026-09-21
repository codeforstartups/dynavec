import json
import logging
from unittest.mock import MagicMock

import pytest

from dynavec.config import DynavecConfig
from dynavec.logging import StructuredJsonFormatter, redact_secrets
from dynavec.stores.dynamodb import DynamoDBStore
from dynavec.stores.s3vectors import S3VectorsStore


def test_redact_secrets():
    payload = {
        "namespace": "tenant_1",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "token": "bearer-xyz",
        "nested": {
            "api_key": "secret-123",
            "safe_count": 42,
            "conn_str": "https://service.com?secret_access_key='secret'",
        },
    }

    sanitized = redact_secrets(payload)

    assert sanitized["namespace"] == "tenant_1"
    assert sanitized["aws_access_key_id"] == "[REDACTED]"
    assert sanitized["aws_secret_access_key"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["safe_count"] == 42
    assert "[REDACTED]" in sanitized["nested"]["conn_str"]


def test_structured_json_formatter():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="dynavec.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="test message",
        args=(),
        exc_info=None,
    )
    record.event_data = {"event": "test.event", "count": 5, "api_key": "supersecret"}

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["logger"] == "dynavec.test"
    assert parsed["level"] == "INFO"
    assert parsed["event"] == "test.event"
    assert parsed["count"] == 5
    assert parsed["api_key"] == "[REDACTED]"
    assert "timestamp" in parsed


def test_dynavec_config_log_level_validation():
    cfg = DynavecConfig(
        vector_bucket="b",
        index="i",
        table="t",
        dimension=8,
        structured_logging=True,
        log_level="DEBUG",
    )
    assert cfg.structured_logging is True
    assert cfg.log_level == "DEBUG"

    with pytest.raises(ValueError, match="log_level must be one of"):
        DynavecConfig(
            vector_bucket="b",
            index="i",
            table="t",
            dimension=8,
            log_level="INVALID",
        )


def test_dynamodb_store_structured_logging(caplog):
    session = MagicMock()
    table = session.resource.return_value.Table.return_value
    table.batch_writer.return_value.__enter__.return_value = MagicMock()
    session.resource.return_value.batch_get_item.return_value = {
        "Responses": {"docs": [{"id": "doc1", "text": "hello", "metadata": {}}]},
        "UnprocessedKeys": {},
    }

    # 1. Opt-in: When False, no structured log emitted
    cfg_disabled = DynavecConfig(
        vector_bucket="b", index="i", table="docs", dimension=8, structured_logging=False
    )
    store_disabled = DynamoDBStore(cfg_disabled, boto_session=session)
    with caplog.at_level(logging.INFO, logger="dynavec.stores.dynamodb"):
        caplog.clear()
        store_disabled.put_many("ns", [("doc1", "hello", {})])
        store_disabled.get_many("ns", ["doc1"])
        store_disabled.delete_many("ns", ["doc1"])
        assert len(caplog.records) == 0

    # 2. When True, structured logs emitted
    cfg_enabled = DynavecConfig(
        vector_bucket="b", index="i", table="docs", dimension=8, structured_logging=True
    )
    store_enabled = DynamoDBStore(cfg_enabled, boto_session=session)
    with caplog.at_level(logging.INFO, logger="dynavec.stores.dynamodb"):
        caplog.clear()
        store_enabled.put_many("ns", [("doc1", "hello", {})])
        store_enabled.get_many("ns", ["doc1"])
        store_enabled.delete_many("ns", ["doc1"])

        assert len(caplog.records) == 3
        events = [json.loads(r.getMessage()) for r in caplog.records]
        assert events[0]["event"] == "dynamodb.put_many"
        assert events[0]["count"] == 1
        assert events[0]["namespace"] == "ns"
        assert "duration_ms" in events[0]

        assert events[1]["event"] == "dynamodb.get_many"
        assert events[1]["requested_count"] == 1
        assert events[1]["returned_count"] == 1

        assert events[2]["event"] == "dynamodb.delete_many"
        assert events[2]["count"] == 1


def test_s3vectors_store_structured_logging(caplog):
    session = MagicMock()
    client = session.client.return_value
    paginator = client.get_paginator.return_value
    paginator.paginate.return_value = [
        {"vectors": [{"key": "ns#1", "data": {"float32": [0.1] * 8}, "distance": 0.05}]}
    ]

    # 1. Opt-in: When False, no logs
    cfg_disabled = DynavecConfig(
        vector_bucket="b", index="i", table="docs", dimension=8, structured_logging=False
    )
    store_disabled = S3VectorsStore(cfg_disabled, boto_session=session)
    with caplog.at_level(logging.INFO, logger="dynavec.stores.s3vectors"):
        caplog.clear()
        store_disabled.put_vectors([("1", [0.1] * 8, {})])
        store_disabled.query([0.1] * 8, top_k=1)
        store_disabled.delete_vectors(["1"])
        assert len(caplog.records) == 0

    # 2. When True, structured logs emitted
    cfg_enabled = DynavecConfig(
        vector_bucket="b", index="i", table="docs", dimension=8, structured_logging=True
    )
    store_enabled = S3VectorsStore(cfg_enabled, boto_session=session)
    with caplog.at_level(logging.INFO, logger="dynavec.stores.s3vectors"):
        caplog.clear()
        store_enabled.put_vectors([("1", [0.1] * 8, {})])
        store_enabled.query([0.1] * 8, top_k=1)
        store_enabled.delete_vectors(["1"])

        assert len(caplog.records) == 3
        events = [json.loads(r.getMessage()) for r in caplog.records]
        assert events[0]["event"] == "s3vectors.put_vectors"
        assert events[0]["count"] == 1
        assert events[0]["bucket"] == "b"

        assert events[1]["event"] == "s3vectors.query"
        assert events[1]["top_k"] == 1
        assert events[1]["returned_count"] == 1

        assert events[2]["event"] == "s3vectors.delete_vectors"
        assert events[2]["count"] == 1
