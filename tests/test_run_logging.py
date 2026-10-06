"""Unit tests for run & node structured logging with secret redaction (Issue #300)."""

from __future__ import annotations

import io
import json
import logging
from unittest.mock import MagicMock

from dynavec import (
    DynamoDBLogSink,
    InMemoryLogSink,
    RunLogger,
    RunLogRecord,
    StdoutSink,
    get_current_correlation,
    get_run_logger,
    log_run_event,
    node_context,
    run_context,
    set_default_run_logger,
)
from dynavec.logging import redact_secrets


def test_redact_secrets_extended():
    """Verify secret redaction across bearer tokens, private keys, and nested structures."""
    sample = {
        "user": "alice",
        "api_key": "sk-1234567890abcdef",
        "auth_token": "secret-token",
        "private_key": "-----BEGIN PRIVATE KEY-----",
        "bearer_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz",
        "nested": {
            "credentials": {"password": "pass123", "normal": "ok"},
            "headers": ["Authorization: Bearer mysecrettoken12345678"],
        },
    }

    redacted = redact_secrets(sample)

    assert redacted["user"] == "alice"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["auth_token"] == "[REDACTED]"
    assert redacted["private_key"] == "[REDACTED]"
    assert "[REDACTED]" in redacted["bearer_header"]
    assert redacted["nested"]["credentials"]["password"] == "[REDACTED]"
    assert redacted["nested"]["credentials"]["normal"] == "ok"
    assert "[REDACTED]" in redacted["nested"]["headers"][0]


def test_run_and_node_context_propagation():
    """Verify that run_context and node_context properly track and clear active correlation IDs."""
    assert get_current_correlation() == {
        "run_id": None,
        "node_id": None,
        "span_id": None,
    }

    with run_context("run-abc-123") as rid:
        assert rid == "run-abc-123"
        assert get_current_correlation()["run_id"] == "run-abc-123"
        assert get_current_correlation()["node_id"] is None

        with node_context("node-llm-1", "span-999") as (nid, sid):
            assert nid == "node-llm-1"
            assert sid == "span-999"
            corr = get_current_correlation()
            assert corr["run_id"] == "run-abc-123"
            assert corr["node_id"] == "node-llm-1"
            assert corr["span_id"] == "span-999"

        # Out of node context
        assert get_current_correlation()["node_id"] is None
        assert get_current_correlation()["span_id"] is None
        assert get_current_correlation()["run_id"] == "run-abc-123"

    # Out of run context
    assert get_current_correlation()["run_id"] is None


def test_context_auto_generated_ids():
    """Verify auto-generated UUIDs when no explicit IDs are provided."""
    with run_context() as rid:
        assert rid.startswith("run_")
        with node_context("retriever_node") as (nid, sid):
            assert nid == "retriever_node"
            assert sid.startswith("span_")


def test_run_logger_with_in_memory_sink():
    """Verify RunLogger emits structured records to InMemoryLogSink with correlation and redaction."""
    sink = InMemoryLogSink()
    logger = RunLogger(sinks=[sink], min_level=logging.DEBUG)

    with run_context("run-100"):
        logger.info("Starting run", tenant="acme", api_key="super_secret")

        with node_context("node-retrieve"):
            logger.debug(
                "Querying vectors",
                query="How to configure AWS?",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            )

        with node_context("node-llm"):
            logger.warning("High latency detected", duration_ms=1520)

    records = sink.records
    assert len(records) == 3

    # First record
    r1 = records[0]
    assert r1.run_id == "run-100"
    assert r1.node_id is None
    assert r1.level == "INFO"
    assert r1.message == "Starting run"
    assert r1.data["tenant"] == "acme"
    assert r1.data["api_key"] == "[REDACTED]"

    # Second record
    r2 = records[1]
    assert r2.run_id == "run-100"
    assert r2.node_id == "node-retrieve"
    assert r2.span_id is not None
    assert r2.level == "DEBUG"
    assert r2.data["aws_access_key_id"] == "[REDACTED]"

    # Filter sink
    llm_records = sink.filter(node_id="node-llm")
    assert len(llm_records) == 1
    assert llm_records[0].level == "WARNING"

    sink.clear()
    assert len(sink.records) == 0


def test_run_logger_min_level_filtering():
    """Verify min_level prevents emitting low-level records."""
    sink = InMemoryLogSink()
    logger = RunLogger(sinks=[sink], min_level=logging.WARNING)

    with run_context("run-200"):
        logger.debug("Debug msg")
        logger.info("Info msg")
        logger.warning("Warning msg")
        logger.error("Error msg")

    assert len(sink.records) == 2
    levels = [r.level for r in sink.records]
    assert levels == ["WARNING", "ERROR"]


def test_run_logger_exception_redaction():
    """Verify exception tracebacks are captured and redacted."""
    sink = InMemoryLogSink()
    logger = RunLogger(sinks=[sink], min_level=logging.INFO)

    try:
        raise ValueError("Failed connecting with token=secret_token_12345")
    except ValueError as e:
        with run_context("run-err"):
            logger.exception("Operation failed", exc=e)

    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.level == "ERROR"
    assert rec.exception is not None
    assert "ValueError" in rec.exception
    assert "[REDACTED]" in rec.exception


def test_stdout_sink():
    """Verify StdoutSink outputs formatted single-line JSON to stream."""
    stream = io.StringIO()
    sink = StdoutSink(stream=stream)
    logger = RunLogger(sinks=[sink], min_level=logging.INFO)

    with run_context("run-stdout"):
        with node_context("node-1"):
            logger.info("Processing step", secret_key="supersecret")

    output = stream.getvalue().strip()
    assert output
    parsed = json.loads(output)
    assert parsed["run_id"] == "run-stdout"
    assert parsed["node_id"] == "node-1"
    assert parsed["level"] == "INFO"
    assert parsed["data"]["secret_key"] == "[REDACTED]"


def test_dynamodb_log_sink():
    """Verify DynamoDBLogSink writes structured items to a DynamoDB table mock."""
    mock_table = MagicMock()
    sink = DynamoDBLogSink(table=mock_table, namespace="tenant_x", buffer_size=1)

    rec = RunLogRecord(
        timestamp="2026-10-06T08:00:00Z",
        run_id="run-ddb-1",
        level="INFO",
        message="Node finished",
        node_id="node-llm",
        span_id="span-123",
        data={"tokens": 150, "token": "secret_abc"},
    )

    sink.emit(rec)

    assert mock_table.put_item.called
    call_kwargs = mock_table.put_item.call_args[1]
    item = call_kwargs["Item"]

    assert item["ns"] == "tenant_x"
    assert item["run_id"] == "run-ddb-1"
    assert item["node_id"] == "node-llm"
    assert item["span_id"] == "span-123"
    assert item["level"] == "INFO"
    assert item["data"]["tokens"] == 150
    assert item["data"]["token"] == "[REDACTED]"
    assert item["pk"].startswith("tenant_x#run_log#run-ddb-1#")


def test_dynamodb_log_sink_buffering():
    """Verify DynamoDBLogSink batch buffering and flush behavior."""
    mock_table = MagicMock()
    batch_writer = MagicMock()
    mock_table.batch_writer.return_value.__enter__.return_value = batch_writer

    sink = DynamoDBLogSink(table=mock_table, namespace="test_ns", buffer_size=3)

    for i in range(2):
        rec = RunLogRecord(
            timestamp=f"2026-10-06T08:00:0{i}Z",
            run_id="run-batch",
            level="INFO",
            message=f"msg {i}",
        )
        sink.emit(rec)

    # Not flushed yet
    assert not batch_writer.put_item.called

    # 3rd record triggers buffer flush
    rec3 = RunLogRecord(
        timestamp="2026-10-06T08:00:03Z",
        run_id="run-batch",
        level="INFO",
        message="msg 3",
    )
    sink.emit(rec3)

    assert batch_writer.put_item.call_count == 3


def test_default_run_logger_and_convenience_function():
    """Verify set_default_run_logger and log_run_event convenience function."""
    mem_sink = InMemoryLogSink()
    custom_logger = RunLogger(sinks=[mem_sink], min_level=logging.INFO)
    old_logger = get_run_logger()

    try:
        set_default_run_logger(custom_logger)
        with run_context("run-convenience"):
            log_run_event("INFO", "Convenience log message", step=1)

        assert len(mem_sink.records) == 1
        assert mem_sink.records[0].run_id == "run-convenience"
        assert mem_sink.records[0].message == "Convenience log message"
    finally:
        set_default_run_logger(old_logger)
