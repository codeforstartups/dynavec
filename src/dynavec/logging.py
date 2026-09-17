"""Structured logging for dynavec store operations and lifecycle events.

Opt-in via DynavecConfig(structured_logging=True). Formats store events as
structured JSON records without leaking AWS credentials, tokens, or private metadata.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

# Sensitive keys whose values must always be redacted.
_SECRET_KEY_PATTERN = re.compile(
    r"(?:secret|password|token|key_id|access_key|auth|credential|api_key)",
    re.IGNORECASE,
)

# Common credential patterns in string values.
_AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")
_AWS_SECRET_PATTERN = re.compile(r"(?i)(secret_access_key\s*[=:]\s*)['\"][^'\"]+['\"]")


def redact_secrets(val: Any, key_name: str | None = None) -> Any:
    """Recursively redact secrets from dictionaries, lists, and strings."""
    if key_name and _SECRET_KEY_PATTERN.search(key_name):
        return "[REDACTED]"

    if isinstance(val, dict):
        return {k: redact_secrets(v, key_name=str(k)) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [redact_secrets(v) for v in val]
    if isinstance(val, str):
        masked = _AWS_ACCESS_KEY_PATTERN.sub("[REDACTED_AWS_KEY]", val)
        return _AWS_SECRET_PATTERN.sub(r"\1'[REDACTED]'", masked)
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
