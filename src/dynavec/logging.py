"""Structured logging for dynavec store operations, runs, and agent execution.

Opt-in via DynavecConfig(structured_logging=True). Formats store events,
agent runs, and node executions as structured JSON records without leaking
AWS credentials, API tokens, passwords, or private metadata.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import traceback
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Sensitive keys whose values must always be redacted.
_SECRET_KEY_PATTERN = re.compile(
    r"(?:secret|password|token|key_id|access_key|auth|credential|api_key|private_key|bearer)",
    re.IGNORECASE,
)

# Keys representing AI token counts/metrics that should NOT be redacted.
_SAFE_METRIC_KEY_PATTERN = re.compile(
    r"^(?:tokens|.*_tokens|token_count|tokens_used|token_usage)$",
    re.IGNORECASE,
)

# Common credential patterns in string values.
_AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")
_AWS_SECRET_PATTERN = re.compile(r"(?i)(secret_access_key\s*[=:]\s*)['\"][^'\"]+['\"]")
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9_\-\.]{15,}\b")
_ASSIGNMENT_SECRET_PATTERN = re.compile(
    r"(?i)\b((?:secret_key|secret|password|token|api_key|access_key)\s*[=:]\s*)(['\"]?[^\s,'\"\]\}]+['\"]?)"
)


def redact_secrets(val: Any, key_name: str | None = None) -> Any:
    """Recursively redact secrets from dictionaries, lists, sets, and strings."""
    if isinstance(val, dict):
        return {k: redact_secrets(v, key_name=str(k)) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [redact_secrets(v) for v in val]
    if isinstance(val, (set, frozenset)):
        return {redact_secrets(v) for v in val}

    if (
        key_name
        and not _SAFE_METRIC_KEY_PATTERN.search(key_name)
        and _SECRET_KEY_PATTERN.search(key_name)
    ):
        return "[REDACTED]"

    if isinstance(val, str):
        masked = _AWS_ACCESS_KEY_PATTERN.sub("[REDACTED_AWS_KEY]", val)
        masked = _AWS_SECRET_PATTERN.sub(r"\1'[REDACTED]'", masked)
        masked = _BEARER_TOKEN_PATTERN.sub(r"\1[REDACTED]", masked)
        masked = _ASSIGNMENT_SECRET_PATTERN.sub(r"\1[REDACTED]", masked)
        return masked
    return val


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_data") and isinstance(record.event_data, dict):
            payload.update(redact_secrets(record.event_data))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def log_store_event(
    logger: logging.Logger,
    event: str,
    enabled: bool,
    *,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """Emit a structured JSON log entry if structured logging is enabled."""
    if not enabled:
        return

    clean_data = redact_secrets({"event": event, **kwargs})
    msg = json.dumps(clean_data, default=str)
    logger.log(level, msg, extra={"event_data": clean_data})


# --- Run & Node Structured Observability (dynalogs) ---

_CURRENT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dynavec_run_id", default=None
)
_CURRENT_NODE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dynavec_node_id", default=None
)
_CURRENT_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dynavec_span_id", default=None
)


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """Context manager to associate log records with an active agent/graph run."""
    actual_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    token = _CURRENT_RUN_ID.set(actual_run_id)
    try:
        yield actual_run_id
    finally:
        _CURRENT_RUN_ID.reset(token)


@contextmanager
def node_context(node_id: str, span_id: str | None = None) -> Iterator[tuple[str, str]]:
    """Context manager to associate log records with a specific node and span execution."""
    actual_span_id = span_id or f"span_{uuid.uuid4().hex[:8]}"
    tok_node = _CURRENT_NODE_ID.set(node_id)
    tok_span = _CURRENT_SPAN_ID.set(actual_span_id)
    try:
        yield node_id, actual_span_id
    finally:
        _CURRENT_NODE_ID.reset(tok_node)
        _CURRENT_SPAN_ID.reset(tok_span)


def get_current_correlation() -> dict[str, str | None]:
    """Retrieve currently active correlation IDs (run_id, node_id, span_id)."""
    return {
        "run_id": _CURRENT_RUN_ID.get(),
        "node_id": _CURRENT_NODE_ID.get(),
        "span_id": _CURRENT_SPAN_ID.get(),
    }


@dataclass
class RunLogRecord:
    """Structured record for an agent run or node execution event."""

    timestamp: str
    run_id: str
    level: str
    message: str
    node_id: str | None = None
    span_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    exception: str | None = None

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        """Convert the record to a dictionary, optionally applying secret redaction."""
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "level": self.level,
            "message": self.message,
        }
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.span_id is not None:
            payload["span_id"] = self.span_id
        if self.data:
            payload["data"] = self.data
        if self.exception is not None:
            payload["exception"] = self.exception

        return redact_secrets(payload) if redact else payload

    def to_json(self, redact: bool = True) -> str:
        """Serialize the record to a single-line JSON string."""
        return json.dumps(self.to_dict(redact=redact), default=str)


class LogSink:
    """Pluggable destination interface for structured run logs."""

    def emit(self, record: RunLogRecord) -> None:
        """Emit a single log record to the destination."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flush any buffered records."""
        pass

    def close(self) -> None:
        """Close resources used by the sink."""
        pass


class StdoutSink(LogSink):
    """Sink that outputs single-line redacted JSON to a stream (default: sys.stdout)."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream

    def emit(self, record: RunLogRecord) -> None:
        line = record.to_json(redact=True)
        target = self.stream if self.stream is not None else sys.stdout
        print(line, file=target)


class InMemoryLogSink(LogSink):
    """Sink that buffers records in memory for assertions, tests, and debugging."""

    def __init__(self) -> None:
        self.records: list[RunLogRecord] = []

    def emit(self, record: RunLogRecord) -> None:
        self.records.append(record)

    def filter(
        self,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        level: str | None = None,
    ) -> list[RunLogRecord]:
        """Filter captured records by run_id, node_id, or log level."""
        res = self.records
        if run_id is not None:
            res = [r for r in res if r.run_id == run_id]
        if node_id is not None:
            res = [r for r in res if r.node_id == node_id]
        if level is not None:
            res = [r for r in res if r.level.upper() == level.upper()]
        return res

    def clear(self) -> None:
        """Clear all buffered records."""
        self.records.clear()


class DynamoDBLogSink(LogSink):
    """Sink that persists structured run and node logs to a DynamoDB table."""

    def __init__(
        self,
        table: Any,
        *,
        namespace: str = "default",
        buffer_size: int = 1,
    ) -> None:
        self.table = table
        self.namespace = namespace
        self.buffer_size = max(1, buffer_size)
        self._buffer: list[dict[str, Any]] = []

    def emit(self, record: RunLogRecord) -> None:
        redacted = record.to_dict(redact=True)
        short_id = uuid.uuid4().hex[:6]
        pk = f"{self.namespace}#run_log#{record.run_id}#{record.timestamp}#{short_id}"
        item: dict[str, Any] = {
            "pk": pk,
            "ns": self.namespace,
            "run_id": record.run_id,
            "timestamp": record.timestamp,
            "level": record.level,
            "message": redacted.get("message", record.message),
        }
        if record.node_id:
            item["node_id"] = record.node_id
        if record.span_id:
            item["span_id"] = record.span_id
        if record.data:
            item["data"] = redacted.get("data", record.data)
        if record.exception:
            item["exception"] = redacted.get("exception", record.exception)

        # Convert types for DynamoDB store compatibility
        from .stores.dynamodb import _to_dynamo

        dynamo_item = _to_dynamo(item)

        if self.buffer_size == 1:
            self.table.put_item(Item=dynamo_item)
        else:
            self._buffer.append(dynamo_item)
            if len(self._buffer) >= self.buffer_size:
                self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.table.batch_writer() as batch:
            for item in self._buffer:
                batch.put_item(Item=item)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


class RunLogger:
    """Structured logger for agent runs and workflow nodes with secret redaction."""

    def __init__(
        self,
        sinks: Sequence[LogSink] | None = None,
        min_level: int | str = logging.INFO,
    ) -> None:
        self.sinks: list[LogSink] = list(sinks) if sinks is not None else [StdoutSink()]
        if isinstance(min_level, str):
            self.min_level = getattr(logging, min_level.upper(), logging.INFO)
        else:
            self.min_level = int(min_level)

    def add_sink(self, sink: LogSink) -> None:
        """Register an additional log sink."""
        self.sinks.append(sink)

    def log(
        self,
        level: int | str,
        message: str,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        span_id: str | None = None,
        data: dict[str, Any] | None = None,
        exception: str | BaseException | None = None,
        **kwargs: Any,
    ) -> RunLogRecord | None:
        """Emit a structured log record with active run and node correlation."""
        num_level = (
            getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else int(level)
        )
        if num_level < self.min_level:
            return None

        eff_run_id = run_id or _CURRENT_RUN_ID.get() or "unknown_run"
        eff_node_id = node_id if node_id is not None else _CURRENT_NODE_ID.get()
        eff_span_id = span_id if span_id is not None else _CURRENT_SPAN_ID.get()

        merged_data = dict(data or {})
        if kwargs:
            merged_data.update(kwargs)

        exc_str: str | None = None
        if exception is not None:
            if isinstance(exception, BaseException):
                exc_str = "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                )
            else:
                exc_str = str(exception)

        level_name = level.upper() if isinstance(level, str) else logging.getLevelName(level)

        record = RunLogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=eff_run_id,
            node_id=eff_node_id,
            span_id=eff_span_id,
            level=level_name,
            message=redact_secrets(message) if isinstance(message, str) else str(message),
            data=redact_secrets(merged_data),
            exception=redact_secrets(exc_str) if exc_str else None,
        )

        for sink in self.sinks:
            sink.emit(record)

        return record

    def debug(self, message: str, **kwargs: Any) -> RunLogRecord | None:
        """Emit a DEBUG-level run log."""
        return self.log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> RunLogRecord | None:
        """Emit an INFO-level run log."""
        return self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> RunLogRecord | None:
        """Emit a WARNING-level run log."""
        return self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> RunLogRecord | None:
        """Emit an ERROR-level run log."""
        return self.log(logging.ERROR, message, **kwargs)

    def exception(
        self,
        message: str,
        exc: BaseException | None = None,
        **kwargs: Any,
    ) -> RunLogRecord | None:
        """Emit an ERROR-level run log with exception traceback."""
        cur_exc = exc or sys.exc_info()[1]
        return self.log(logging.ERROR, message, exception=cur_exc, **kwargs)


_DEFAULT_RUN_LOGGER: RunLogger = RunLogger()


def get_run_logger() -> RunLogger:
    """Return the global default RunLogger."""
    return _DEFAULT_RUN_LOGGER


def set_default_run_logger(logger: RunLogger) -> None:
    """Set the global default RunLogger."""
    global _DEFAULT_RUN_LOGGER
    _DEFAULT_RUN_LOGGER = logger


def log_run_event(
    level: int | str,
    message: str,
    **kwargs: Any,
) -> RunLogRecord | None:
    """Convenience function to log a run event using the default RunLogger."""
    return get_run_logger().log(level, message, **kwargs)
