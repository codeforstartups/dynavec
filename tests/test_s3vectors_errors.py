"""Offline coverage of S3 Vectors exception translation and retry boundaries."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError

import dynavec
from dynavec import AccessDeniedError, DynavecError, NotFoundError, ValidationError
from dynavec.config import DynavecConfig
from dynavec.stores._s3vectors_errors import translate_client_errors
from dynavec.stores.async_s3vectors import AsyncS3VectorsStore
from dynavec.stores.s3vectors import S3VectorsStore

MAPPINGS = [
    ("NotFoundException", NotFoundError),
    ("AccessDeniedException", AccessDeniedError),
    ("ValidationException", ValidationError),
]


def _client_error(code, message="AWS request failed"):
    error = {"Code": code}
    if message is not None:
        error["Message"] = message
    return ClientError({"Error": error}, "S3VectorsOperation")


def _config():
    return DynavecConfig(vector_bucket="bucket", index="index", table="table", dimension=2)


@pytest.fixture
def sync_store():
    session = MagicMock()
    store = S3VectorsStore(_config(), session)
    return store, session.client.return_value


def _async_store():
    client = MagicMock()
    client.put_vectors = AsyncMock()
    client.get_vectors = AsyncMock()
    session = MagicMock()
    session.client.return_value.__aenter__.return_value = client
    return AsyncS3VectorsStore(_config(), session), client


@pytest.mark.parametrize("code,error_type", MAPPINGS)
def test_maps_selected_errors_with_message_and_cause(code, error_type):
    original = _client_error(code, "Detailed AWS error message")

    with pytest.raises(error_type, match="Detailed AWS error message") as caught:
        with translate_client_errors():
            raise original

    assert caught.value.__cause__ is original


@pytest.mark.parametrize("code,error_type", MAPPINGS)
def test_missing_message_has_stable_fallback(code, error_type):
    original = _client_error(code, message=None)

    with pytest.raises(error_type) as caught:
        with translate_client_errors():
            raise original

    assert str(caught.value) == f"S3 Vectors request failed with {code}."
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("code", ["UnknownException", "ThrottlingException", "InternalServerError"])
def test_unmapped_errors_remain_the_same_object(code):
    original = _client_error(code)

    with pytest.raises(ClientError) as caught:
        with translate_client_errors():
            raise original

    assert caught.value is original


@pytest.mark.parametrize("response", [{}, {"Error": None}, {"Error": {}}, {"Error": {"Code": []}}])
def test_malformed_error_details_are_not_translated(response):
    original = _client_error("UnknownException")
    original.response = response

    with pytest.raises(ClientError) as caught:
        with translate_client_errors():
            raise original

    assert caught.value is original


def test_unrelated_python_error_remains_unchanged():
    original = ValueError("invalid local input")

    with pytest.raises(ValueError) as caught:
        with translate_client_errors():
            raise original

    assert caught.value is original


@pytest.mark.parametrize("error_type", [AccessDeniedError, ValidationError])
def test_new_exceptions_are_public(error_type):
    assert getattr(dynavec, error_type.__name__) is error_type
    assert error_type.__name__ in dynavec.__all__
    assert issubclass(error_type, DynavecError)


@pytest.mark.parametrize("code,error_type", MAPPINGS)
@pytest.mark.parametrize(
    "operation,args,aws_operation",
    [
        ("get_index", (), "get_index"),
        ("_put_batch", ([],), "put_vectors"),
        ("get_vectors", (["key"],), "get_vectors"),
        ("delete_vectors", (["key"],), "delete_vectors"),
    ],
)
def test_sync_direct_operations_translate(
    sync_store, code, error_type, operation, args, aws_operation
):
    store, client = sync_store
    original = _client_error(code)
    aws_call = getattr(client, aws_operation)
    aws_call.side_effect = original

    with pytest.raises(error_type, match="AWS request failed") as caught:
        getattr(store, operation)(*args)

    assert caught.value.__cause__ is original
    aws_call.assert_called_once()


@pytest.mark.parametrize("code,error_type", MAPPINGS)
@pytest.mark.parametrize(
    "operation,args,aws_operation",
    [("_put_batch", ([],), "put_vectors"), ("get_vectors", (["key"],), "get_vectors")],
)
async def test_async_direct_operations_translate(code, error_type, operation, args, aws_operation):
    store, client = _async_store()
    original = _client_error(code)
    aws_call = getattr(client, aws_operation)
    aws_call.side_effect = original

    async with store:
        with pytest.raises(error_type, match="AWS request failed") as caught:
            await getattr(store, operation)(*args)

    assert caught.value.__cause__ is original
    aws_call.assert_awaited_once()


def _failing_pages(original):
    yield {"vectors": [{"key": "first"}], "NextToken": "next"}
    raise original


async def _async_pages(original=None):
    yield {"vectors": [{"key": "first"}], "NextToken": "next"}
    if original is not None:
        raise original


@pytest.mark.parametrize("operation,args", [("query_pages", ([0.1, 0.2], 2)), ("list_pages", ())])
def test_sync_pagination_translates_later_page_errors(sync_store, operation, args):
    store, client = sync_store
    original = _client_error("NotFoundException")
    client.get_paginator.return_value.paginate.return_value = _failing_pages(original)

    pages = getattr(store, operation)(*args)
    client.get_paginator.assert_not_called()
    assert next(pages) == [{"key": "first"}]
    with pytest.raises(NotFoundError) as caught:
        next(pages)

    assert caught.value.__cause__ is original


async def test_async_pagination_translates_later_page_errors():
    store, client = _async_store()
    original = _client_error("NotFoundException")
    client.get_paginator.return_value.paginate.return_value = _async_pages(original)

    async with store:
        pages = store.query_pages([0.1, 0.2], 2)
        client.get_paginator.assert_not_called()
        assert await pages.__anext__() == [{"key": "first"}]
        with pytest.raises(NotFoundError) as caught:
            await pages.__anext__()

    assert caught.value.__cause__ is original


@pytest.mark.parametrize("operation,args", [("query", ([0.1, 0.2], 2)), ("list_pages", ())])
@pytest.mark.parametrize("boundary", ["get_paginator", "paginate"])
def test_sync_paginator_setup_errors_translate(sync_store, operation, args, boundary):
    store, client = sync_store
    original = _client_error("ValidationException")
    call = (
        client.get_paginator
        if boundary == "get_paginator"
        else client.get_paginator.return_value.paginate
    )
    call.side_effect = original

    with pytest.raises(ValidationError) as caught:
        list(getattr(store, operation)(*args))

    assert caught.value.__cause__ is original
    call.assert_called_once()


@pytest.mark.parametrize("boundary", ["get_paginator", "paginate"])
async def test_async_paginator_setup_errors_translate(boundary):
    store, client = _async_store()
    original = _client_error("ValidationException")
    call = (
        client.get_paginator
        if boundary == "get_paginator"
        else client.get_paginator.return_value.paginate
    )
    call.side_effect = original

    async with store:
        with pytest.raises(ValidationError) as caught:
            await store.query([0.1, 0.2], 2)

    assert caught.value.__cause__ is original
    call.assert_called_once()


def test_sync_unknown_error_is_not_retried(sync_store):
    store, client = sync_store
    original = _client_error("UnknownException")
    client.get_vectors.side_effect = original

    with pytest.raises(ClientError) as caught:
        store.get_vectors(["key"])

    assert caught.value is original
    client.get_vectors.assert_called_once()


async def test_async_unknown_error_is_not_retried():
    store, client = _async_store()
    original = _client_error("UnknownException")
    client.get_vectors.side_effect = original

    async with store:
        with pytest.raises(ClientError) as caught:
            await store.get_vectors(["key"])

    assert caught.value is original
    client.get_vectors.assert_awaited_once()


@pytest.mark.parametrize("code", ["ThrottlingException", "InternalServerError"])
def test_sync_batch_retries_then_succeeds(sync_store, monkeypatch, code):
    store, client = sync_store
    client.put_vectors.side_effect = [_client_error(code), {}]
    sleep = MagicMock()
    monkeypatch.setattr("dynavec.utils.time.sleep", sleep)

    store._put_batch([])

    assert client.put_vectors.call_count == 2
    sleep.assert_called_once()


@pytest.mark.parametrize("code", ["ThrottlingException", "InternalServerError"])
async def test_async_batch_retries_then_succeeds(monkeypatch, code):
    store, client = _async_store()
    client.put_vectors.side_effect = [_client_error(code), {}]
    sleep = AsyncMock()
    monkeypatch.setattr("dynavec.utils.asyncio.sleep", sleep)

    async with store:
        await store._put_batch([])

    assert client.put_vectors.await_count == 2
    sleep.assert_awaited_once()


def test_sync_query_retries_page_iteration_failure(sync_store, monkeypatch):
    store, client = sync_store
    client.get_paginator.return_value.paginate.side_effect = [
        _failing_pages(_client_error("ThrottlingException")),
        [{"vectors": [{"key": "success"}]}],
    ]
    sleep = MagicMock()
    monkeypatch.setattr("dynavec.utils.time.sleep", sleep)

    assert store.query([0.1, 0.2], 2) == [{"key": "success"}]

    assert client.get_paginator.call_count == 2
    sleep.assert_called_once()


async def test_async_query_retries_page_iteration_failure(monkeypatch):
    store, client = _async_store()
    client.get_paginator.return_value.paginate.side_effect = [
        _async_pages(_client_error("ThrottlingException")),
        _async_pages(),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr("dynavec.utils.asyncio.sleep", sleep)

    async with store:
        assert await store.query([0.1, 0.2], 2) == [{"key": "first"}]

    assert client.get_paginator.call_count == 2
    sleep.assert_awaited_once()


@pytest.mark.parametrize(
    "operation,args,aws_operation",
    [
        ("get_index", (), "get_index"),
        ("delete_vectors", (["key"],), "delete_vectors"),
        ("list_pages", (), "get_paginator"),
    ],
)
def test_sync_non_retrying_operations_still_do_not_retry(
    sync_store, operation, args, aws_operation
):
    store, client = sync_store
    original = _client_error("ThrottlingException")
    aws_call = getattr(client, aws_operation)
    aws_call.side_effect = original

    with pytest.raises(ClientError) as caught:
        result = getattr(store, operation)(*args)
        if operation == "list_pages":
            list(result)

    assert caught.value is original
    aws_call.assert_called_once()


def test_sync_worker_propagates_mapped_error(sync_store):
    store, client = sync_store
    original = _client_error("AccessDeniedException")

    def upload(**kwargs):
        if kwargs["vectors"][0]["key"] == "key-0":
            raise original

    client.put_vectors.side_effect = upload
    vectors = [(f"key-{i}", [0.1, 0.2], {}) for i in range(501)]

    with pytest.raises(AccessDeniedError) as caught:
        store.put_vectors(vectors, max_workers=2)

    assert caught.value.__cause__ is original
    assert client.put_vectors.call_count == 2


async def test_async_upload_translates_and_cleans_up_pending_tasks():
    store, client = _async_store()
    context = store._session.client.return_value
    original = _client_error("AccessDeniedException")
    sibling_started = asyncio.Event()
    sibling_cleaned_up = asyncio.Event()
    tasks = []

    async def upload(**kwargs):
        tasks.append(asyncio.current_task())
        if kwargs["vectors"][0]["key"] == "key-0":
            await sibling_started.wait()
            raise original
        sibling_started.set()
        try:
            await asyncio.Future()
        finally:
            sibling_cleaned_up.set()

    client.put_vectors.side_effect = upload
    vectors = [(f"key-{i}", [0.1, 0.2], {}) for i in range(501)]

    with pytest.raises(AccessDeniedError) as caught:
        async with store:
            await asyncio.wait_for(store.put_vectors(vectors, max_workers=2), timeout=2)

    assert caught.value.__cause__ is original
    assert sibling_cleaned_up.is_set()
    assert len(tasks) == 2
    assert all(task.done() for task in tasks)
    assert sum(task.cancelled() for task in tasks) == 1
    assert store._client is None
    assert store._client_context is None
    context.__aexit__.assert_awaited_once()
