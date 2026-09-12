from decimal import Decimal
from unittest.mock import MagicMock, call

from dynavec.config import DynavecConfig
from dynavec.stores.dynamodb import DynamoDBStore


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
