from dynavec.client import Dynavec
from dynavec.graph import GraphStore
from dynavec.stores.dynamodb import DynamoDBStore


def test_document_keys_escape_components_without_collisions():
    first = DynamoDBStore._pk("tenant#one", "doc")
    second = DynamoDBStore._pk("tenant", "one#doc")

    assert first == "tenant%23one#doc"
    assert second == "tenant#one%23doc"
    assert first != second


def test_s3_key_round_trips_escaped_components():
    key = Dynavec._s3_key(None, "tenant#one", "doc%two")

    assert key == "tenant%23one#doc%25two"
    assert Dynavec._split_key(None, key) == ("tenant#one", "doc%two")


def test_graph_keys_escape_namespace_and_entity_id():
    assert GraphStore._node_pk("tenant#one", "entity#two") == "tenant%23one#node#entity%23two"
