"""Tests for the async DynamoDB store."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from dynavec.config import DynavecConfig
from dynavec.exceptions import ItemTooLargeError
from dynavec.stores.async_dynamodb import AsyncDynamoDBStore
from dynavec.stores.dynamodb import MAX_ITEM_BYTES


def _config() -> DynavecConfig:
    return DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="docs",
        dimension=8,
    )


class FakeBatchWriter:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeBatchWriter:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def put_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs["Item"])


class FakeTable:
    def __init__(self) -> None:
        self.writer = FakeBatchWriter()
        self.batch_writer_calls: list[dict[str, Any]] = []

    def batch_writer(self, **kwargs: Any) -> FakeBatchWriter:
        self.batch_writer_calls.append(kwargs)
        return self.writer


class FakeDynamoResource:
    def __init__(
        self,
        table: FakeTable,
        responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.table = table
        self.responses = list(responses or [])
        self.batch_get_calls: list[dict[str, Any]] = []
        self.table_calls: list[str] = []

    async def Table(self, name: str) -> FakeTable:
        self.table_calls.append(name)
        return self.table

    async def batch_get_item(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.batch_get_calls.append(kwargs)

        if self.responses:
            return self.responses.pop(0)

        return {
            "Responses": {"docs": []},
            "UnprocessedKeys": {},
        }


class FakeResourceContext:
    def __init__(self, resource: FakeDynamoResource) -> None:
        self.resource = resource
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeDynamoResource:
        self.entered = True
        return self.resource

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True


class FakeSession:
    def __init__(self, resource: FakeDynamoResource) -> None:
        self.resource_instance = resource
        self.context: FakeResourceContext | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def resource(
        self,
        service_name: str,
        **kwargs: Any,
    ) -> FakeResourceContext:
        self.calls.append((service_name, kwargs))
        self.context = FakeResourceContext(self.resource_instance)
        return self.context


async def test_async_store_enters_and_closes_resource():
    table = FakeTable()
    resource = FakeDynamoResource(table)
    session = FakeSession(resource)

    async with AsyncDynamoDBStore(_config(), session):
        assert session.context is not None
        assert session.context.entered
        assert resource.table_calls == ["docs"]

    assert session.context is not None
    assert session.context.exited
    assert session.calls[0][0] == "dynamodb"


async def test_put_many_builds_and_writes_items():
    table = FakeTable()
    resource = FakeDynamoResource(table)
    session = FakeSession(resource)

    async with AsyncDynamoDBStore(_config(), session) as store:
        await store.put_many(
            "tenant",
            [
                ("a", "First", {"score": 0.5}),
                ("b", "Second", {"year": 2026}),
            ],
        )

    assert table.batch_writer_calls == [{"overwrite_by_pkeys": ["pk"]}]

    assert len(table.writer.items) == 2

    assert table.writer.items[0]["pk"] == "tenant#a"
    assert table.writer.items[0]["id"] == "a"
    assert table.writer.items[0]["text"] == "First"
    assert table.writer.items[0]["metadata"]["score"] == Decimal("0.5")


async def test_put_many_writes_custom_ttl_attribute():
    config = DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="docs",
        dimension=8,
        dynamodb_ttl_attribute="expires_at",
    )

    table = FakeTable()
    resource = FakeDynamoResource(table)
    session = FakeSession(resource)

    async with AsyncDynamoDBStore(config, session) as store:
        await store.put_many(
            "tenant",
            [
                (
                    "a",
                    "First",
                    {
                        "topic": "test",
                        "_ttl": 1_700_000_060,
                    },
                )
            ],
        )

    item = table.writer.items[0]

    assert item["expires_at"] == 1_700_000_060
    assert "_ttl" not in item["metadata"]
    assert item["metadata"]["topic"] == "test"


async def test_put_many_rejects_oversized_item_before_writing():
    table = FakeTable()
    resource = FakeDynamoResource(table)
    session = FakeSession(resource)

    async with AsyncDynamoDBStore(_config(), session) as store:
        with pytest.raises(ItemTooLargeError):
            await store.put_many(
                "tenant",
                [
                    ("ok", "fits", {}),
                    (
                        "big",
                        "x" * MAX_ITEM_BYTES,
                        {},
                    ),
                ],
            )

    assert table.batch_writer_calls == []


async def test_get_many_retries_only_unprocessed_keys():
    table = FakeTable()

    remaining = {
        "docs": {
            "Keys": [
                {"pk": "tenant#b"},
                {"pk": "tenant#c"},
            ]
        }
    }

    resource = FakeDynamoResource(
        table,
        responses=[
            {
                "Responses": {
                    "docs": [
                        {
                            "id": "a",
                            "text": "First document",
                            "metadata": {
                                "topic": "aws",
                            },
                        }
                    ]
                },
                "UnprocessedKeys": remaining,
            },
            {
                "Responses": {
                    "docs": [
                        {
                            "id": "c",
                            "text": "Third document",
                            "metadata": {
                                "score": Decimal("0.1"),
                            },
                        },
                        {
                            "id": "b",
                            "text": "Second document",
                            "metadata": {
                                "year": Decimal("2026"),
                            },
                        },
                    ]
                },
                "UnprocessedKeys": {},
            },
        ],
    )

    session = FakeSession(resource)

    async with AsyncDynamoDBStore(_config(), session) as store:
        documents = await store.get_many(
            "tenant",
            ["a", "b", "c"],
        )

    assert resource.batch_get_calls == [
        {
            "RequestItems": {
                "docs": {
                    "Keys": [
                        {"pk": "tenant#a"},
                        {"pk": "tenant#b"},
                        {"pk": "tenant#c"},
                    ]
                }
            }
        },
        {
            "RequestItems": remaining,
        },
    ]

    assert documents == {
        "a": {
            "text": "First document",
            "metadata": {"topic": "aws"},
        },
        "b": {
            "text": "Second document",
            "metadata": {"year": 2026},
        },
        "c": {
            "text": "Third document",
            "metadata": {"score": 0.1},
        },
    }


async def test_get_many_batches_at_100_keys():
    table = FakeTable()

    resource = FakeDynamoResource(
        table,
        responses=[
            {
                "Responses": {"docs": []},
                "UnprocessedKeys": {},
            },
            {
                "Responses": {"docs": []},
                "UnprocessedKeys": {},
            },
            {
                "Responses": {"docs": []},
                "UnprocessedKeys": {},
            },
        ],
    )

    session = FakeSession(resource)
    ids = [f"doc-{i}" for i in range(250)]

    async with AsyncDynamoDBStore(_config(), session) as store:
        await store.get_many("tenant", ids)

    sizes = [len(call["RequestItems"]["docs"]["Keys"]) for call in resource.batch_get_calls]

    assert sizes == [100, 100, 50]


async def test_operations_require_open_store():
    store = AsyncDynamoDBStore(
        _config(),
        FakeSession(FakeDynamoResource(FakeTable())),
    )

    with pytest.raises(RuntimeError, match="not open"):
        await store.put_many(
            "tenant",
            [("a", "text", {})],
        )

    with pytest.raises(RuntimeError, match="not open"):
        await store.get_many(
            "tenant",
            ["a"],
        )


async def test_get_many_returns_ttl():
    config = DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="docs",
        dimension=8,
        dynamodb_ttl_attribute="expires_at",
    )

    table = FakeTable()
    resource = FakeDynamoResource(
        table,
        responses=[
            {
                "Responses": {
                    "docs": [
                        {
                            "id": "a",
                            "text": "First",
                            "metadata": {
                                "topic": "test",
                            },
                            "expires_at": Decimal("1700000060"),
                        }
                    ]
                },
                "UnprocessedKeys": {},
            }
        ],
    )

    session = FakeSession(resource)

    async with AsyncDynamoDBStore(config, session) as store:
        documents = await store.get_many(
            "tenant",
            ["a"],
        )

    assert documents["a"]["ttl"] == 1_700_000_060


async def test_enter_closes_resource_if_table_creation_fails():
    table = FakeTable()
    resource = FakeDynamoResource(table)

    async def fail_table(name: str):
        raise RuntimeError("table creation failed")

    resource.Table = fail_table  # type: ignore[method-assign]
    session = FakeSession(resource)

    store = AsyncDynamoDBStore(_config(), session)

    with pytest.raises(
        RuntimeError,
        match="table creation failed",
    ):
        await store.__aenter__()

    assert session.context is not None
    assert session.context.exited
    assert store._ddb is None
    assert store._table is None
    assert store._resource_context is None
