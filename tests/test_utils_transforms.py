"""Tests for decorators/generators and the transform pipeline (pure, no AWS)."""

from unittest.mock import patch

import pytest

from dynavec.transforms import TransformContext, TransformPipeline, as_pipeline
from dynavec.utils import TokenBucket, async_retry, chunked, is_retryable, retry


def test_chunked_generator():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunked([], 3)) == []
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_retry_retries_then_succeeds():
    calls = {"n": 0}

    class Throttle(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    @retry(max_attempts=5, base_delay=0.0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Throttle()
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_non_retryable():
    calls = {"n": 0}

    @retry(max_attempts=5, base_delay=0.0)
    def boom():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    assert calls["n"] == 1  # not retried


def test_is_retryable_detects_codes():
    class E(Exception):
        response = {"Error": {"Code": "ProvisionedThroughputExceededException"}}

    assert is_retryable(E())
    assert not is_retryable(ValueError("x"))


def test_transform_pipeline_composes():
    def upper(ctx: TransformContext) -> TransformContext:
        ctx.text = (ctx.text or "").upper()
        return ctx

    def tag(ctx: TransformContext) -> TransformContext:
        ctx.metadata["stage"] = "processed"
        return ctx

    pipe = TransformPipeline([upper, tag])
    out = pipe(TransformContext(id="1", text="hi", metadata={}))
    assert out.text == "HI"
    assert out.metadata["stage"] == "processed"
    assert len(pipe) == 2


def test_as_pipeline_coercions():
    assert as_pipeline(None) is None
    single = as_pipeline(lambda c: c)
    assert isinstance(single, TransformPipeline) and len(single) == 1
    multi = as_pipeline([lambda c: c, lambda c: c])
    assert len(multi) == 2


def test_retry_uses_retry_delay(monkeypatch):
    calls = {"n": 0}
    delays = []

    class Throttle(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    monkeypatch.setattr("dynavec.utils.time.sleep", delays.append)

    @retry(
        max_attempts=3,
        retry_delay=lambda exc: 3.0,
    )
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise Throttle()
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2
    assert delays == [3.0]


def test_token_bucket_consumes_tokens():
    with patch(
        "dynavec.utils.time.monotonic",
        side_effect=[100.0, 100.0, 100.0],
    ):
        bucket = TokenBucket(rate=10, capacity=2)
        assert bucket.tokens == 2

        bucket.acquire()
        assert bucket.tokens == 1

        bucket.acquire()
        assert bucket.tokens == 0


def test_token_bucket_refills_tokens():
    with patch(
        "dynavec.utils.time.monotonic",
        side_effect=[100.0, 100.0, 100.0, 100.2],
    ):
        bucket = TokenBucket(rate=10, capacity=2)

        bucket.acquire()
        bucket.acquire()

        # 0.2 seconds * 10 tokens/sec = 2 new tokens
        bucket.acquire()

        assert bucket.tokens == 1


def test_token_bucket_waits_for_token():
    with (
        patch(
            "dynavec.utils.time.monotonic",
            side_effect=[100.0, 100.0, 100.0, 100.05, 100.11],
        ),
        patch("dynavec.utils.time.sleep") as sleep,
    ):
        bucket = TokenBucket(rate=10, capacity=2)

        bucket.acquire()
        bucket.acquire()
        bucket.acquire()

        sleep.assert_called_once_with(pytest.approx(0.05))
        assert bucket.tokens == pytest.approx(0.1)


def test_token_bucket_does_not_exceed_capacity():
    with patch(
        "dynavec.utils.time.monotonic",
        side_effect=[100.0, 100.0, 101.0],
    ):
        bucket = TokenBucket(rate=10, capacity=2)
        assert bucket.tokens == 2

        bucket.acquire()
        assert bucket.tokens == 1

        bucket.acquire()
        assert bucket.tokens == 1


async def test_token_bucket_async_waits_for_token(
    monkeypatch,
):
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(
        "dynavec.utils.asyncio.sleep",
        fake_sleep,
    )

    with patch(
        "dynavec.utils.time.monotonic",
        side_effect=[
            100.0,
            100.0,
            100.0,
            100.05,
            100.11,
        ],
    ):
        bucket = TokenBucket(
            rate=10,
            capacity=2,
        )

        await bucket.acquire_async()
        await bucket.acquire_async()
        await bucket.acquire_async()

    assert delays == [pytest.approx(0.05)]
    assert bucket.tokens == pytest.approx(0.1)


async def test_async_retry_retries_then_succeeds():
    calls = {"n": 0}

    class Throttle(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    @async_retry(max_attempts=5, base_delay=0.0)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Throttle()
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


async def test_async_retry_does_not_retry_non_retryable():
    calls = {"n": 0}

    @async_retry(max_attempts=5, base_delay=0.0)
    async def boom():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await boom()

    assert calls["n"] == 1


async def test_async_retry_uses_retry_delay(monkeypatch):
    calls = {"n": 0}
    delays = []

    class Throttle(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("dynavec.utils.asyncio.sleep", fake_sleep)

    @async_retry(
        max_attempts=3,
        retry_delay=lambda exc: 3.0,
    )
    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise Throttle()
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 2
    assert delays == [3.0]
