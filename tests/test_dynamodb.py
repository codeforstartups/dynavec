from decimal import Decimal
from unittest.mock import MagicMock, call

import pytest

from dynavec.config import DynavecConfig
from dynavec.exceptions import ItemTooLargeError
from dynavec.stores.dynamodb import (
    MAX_ITEM_BYTES,
    DynamoDBStore,
    _build_item,
    check_item_size,
    item_size_bytes,
)


def test_get_many_retries_only_unprocessed_keys_and_hydrates_all_documents():
    config = DynavecConfig(vector_bucket="bucket", index="index", table="docs", dimension=8)
    session = MagicMock()
    ddb = session.resource.return_value
    remaining = {"docs": {"Keys": [{"pk": "tenant#b"}, {"pk": "tenant#c"}]}}
    ddb.batch_get_item.side_effect = [
        {
            "Responses": {
                "docs": [{"id": "a", "text": "First document", "metadata": {"topic": "aws"}}]
            },
            "UnprocessedKeys": remaining,
        },
        {
            "Responses": {
                "docs": [
                    {"id": "c", "text": "Third document", "metadata": {"score": Decimal("0.1")}},
                    {"id": "b", "text": "Second document", "metadata": {"year": Decimal("2026")}},
                ]
            },
            "UnprocessedKeys": {},
        },
    ]
    store = DynamoDBStore(config, boto_session=session)

    documents = store.get_many("tenant", ["a", "b", "c"])

    assert ddb.batch_get_item.call_args_list == [
        call(
            RequestItems={
                "docs": {"Keys": [{"pk": "tenant#a"}, {"pk": "tenant#b"}, {"pk": "tenant#c"}]}
            }
        ),
        call(RequestItems=remaining),
    ]
    assert documents == {
        "a": {"text": "First document", "metadata": {"topic": "aws"}},
        "b": {"text": "Second document", "metadata": {"year": 2026}},
        "c": {"text": "Third document", "metadata": {"score": 0.1}},
    }


# ------------------------------------------------------------------ item size
def test_item_size_follows_dynamodb_sizing_rules():
    assert item_size_bytes({"pk": "a#b"}) == 2 + 3
    assert item_size_bytes({"s": "é"}) == 1 + 2  # UTF-8 bytes, not characters
    assert item_size_bytes({"n": Decimal("12345")}) == 1 + 4  # 3 digit bytes + 1
    assert item_size_bytes({"n": Decimal("1.500")}) == 1 + 2  # trailing zeros trimmed
    assert item_size_bytes({"b": True, "z": None}) == (1 + 1) + (1 + 1)
    # map: 3 overhead + per entry (1 + key + value); list: 3 overhead + per element (1 + value)
    assert item_size_bytes({"m": {"ab": "xyz"}}) == 1 + 3 + (1 + 2 + 3)
    assert item_size_bytes({"l": ["xy", Decimal("7")]}) == 1 + 3 + (1 + 2) + (1 + 2)


def test_check_item_size_allows_exactly_the_limit():
    base = item_size_bytes(_build_item("tenant", "doc", "", {"topic": "aws"}))
    fits = "x" * (MAX_ITEM_BYTES - base)

    check_item_size("tenant", "doc", fits, {"topic": "aws"})

    with pytest.raises(ItemTooLargeError) as info:
        check_item_size("tenant", "doc", fits + "x", {"topic": "aws"})
    assert info.value.size_bytes == MAX_ITEM_BYTES + 1
    assert (info.value.doc_id, info.value.namespace) == ("doc", "tenant")


def test_put_many_rejects_oversized_item_without_writing():
    config = DynavecConfig(vector_bucket="bucket", index="index", table="docs", dimension=8)
    session = MagicMock()
    table = session.resource.return_value.Table.return_value
    store = DynamoDBStore(config, boto_session=session)

    with pytest.raises(ItemTooLargeError):
        store.put_many("tenant", [("ok", "fits", {}), ("big", "x" * MAX_ITEM_BYTES, {})])

    table.batch_writer.assert_not_called()
