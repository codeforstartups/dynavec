"""Tests for decorators/generators and the transform pipeline (pure, no AWS)."""

import pytest

from dynavec.transforms import TransformContext, TransformPipeline, as_pipeline
from dynavec.utils import chunked, is_retryable, retry


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
