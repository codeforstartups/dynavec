"""Shared error translation for S3 Vectors AWS calls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from botocore.exceptions import ClientError

from ..exceptions import AccessDeniedError, DynavecError, NotFoundError, ValidationError

_ERROR_TYPES: dict[str, type[DynavecError]] = {
    "NotFoundException": NotFoundError,
    "AccessDeniedException": AccessDeniedError,
    "ValidationException": ValidationError,
}


@contextmanager
def translate_client_errors() -> Iterator[None]:
    """Translate selected permanent errors, leaving retryable errors intact."""
    try:
        yield
    except ClientError as exc:
        response = exc.response
        error = response.get("Error") if isinstance(response, Mapping) else None
        if not isinstance(error, Mapping):
            raise
        code = error.get("Code")
        error_type = _ERROR_TYPES.get(code) if isinstance(code, str) else None
        if error_type is None:
            raise
        message = error.get("Message")
        if not isinstance(message, str) or not message:
            message = f"S3 Vectors request failed with {code}."
        raise error_type(message) from exc
