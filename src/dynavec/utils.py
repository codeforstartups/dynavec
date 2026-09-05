"""Small Pythonic building blocks: decorators, closures, generators.

These keep the rest of the codebase terse and make dynavec pleasant to embed in
agent frameworks (clear retries, optional latency hooks, lazy batching).
"""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

# botocore error codes that are safe to retry (throttling / transient).
_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
        "RequestLimitExceeded",
        "InternalServerError",
        "ServiceUnavailable",
        "TransactionInProgressException",
        "SlowDown",
    }
)


def is_retryable(exc: Exception) -> bool:
    """Duck-typed check for a retryable AWS error (no botocore import needed)."""
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return code in _RETRYABLE_CODES


def retry(
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    retry_on: Callable[[Exception], bool] = is_retryable,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory: exponential backoff with full jitter.

    A closure over the retry policy — the classic decorator-with-arguments
    pattern. Only retries when ``retry_on(exc)`` is True; re-raises otherwise.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    attempt += 1
                    if attempt >= max_attempts or not retry_on(exc):
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    time.sleep(random.uniform(0, delay))  # full jitter

        return wrapper

    return decorator


def timed(sink: Optional[Callable[[str, float], None]] = None):
    """Decorator: report wall-clock seconds to ``sink(name, seconds)``.

    Handy for wiring dynavec latency into an agent's tracing/telemetry.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if sink is not None:
                    sink(fn.__qualname__, elapsed)

        return wrapper

    return decorator


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Lazily yield fixed-size chunks (a generator, so memory stays flat)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
