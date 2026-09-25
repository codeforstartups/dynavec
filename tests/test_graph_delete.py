"""Deleting graph nodes and edges.

Runs the real ``GraphStore`` against a moto DynamoDB table rather than a dict
fake: the behaviour under test is the conditional read-filter-write on the
embedded adjacency list, which only a real (emulated) table can check.
"""

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore


@pytest.fixture
def store():
    with mock_aws():
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )
        session.client("dynamodb").create_table(
            TableName="test-table",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        config = DynavecConfig(
            vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
        )
        yield GraphStore(config, boto_session=session)


def _edges(store, ns, entity_id):
    return [(e["relation"], e["target"]) for e in store.get_node(ns, entity_id)["edges"]]


# ----------------------------------------------------------------- delete_edge
def test_delete_edge_removes_only_the_matching_edge(store):
    store.add_edge("ns", "a", "knows", "b")
    store.add_edge("ns", "a", "owns", "b")  # same target, other relation
    store.add_edge("ns", "a", "knows", "c")  # same relation, other target

    assert store.delete_edge("ns", "a", "knows", "b") == 1

    assert _edges(store, "ns", "a") == [("owns", "b"), ("knows", "c")]
    assert store.get_node("ns", "b") is not None  # endpoints survive


def test_delete_edge_removes_duplicate_entries(store):
    store.add_edge("ns", "a", "knows", "b")
    store.add_edge("ns", "a", "knows", "b")

    assert store.delete_edge("ns", "a", "knows", "b") == 2
    assert _edges(store, "ns", "a") == []


def test_delete_edge_is_idempotent(store):
    store.add_edge("ns", "a", "knows", "b")

    assert store.delete_edge("ns", "a", "knows", "b") == 1
    assert store.delete_edge("ns", "a", "knows", "b") == 0
    assert store.delete_edge("ns", "missing", "knows", "b") == 0
    assert store.get_node("ns", "missing") is None  # a no-op never creates a node


def test_delete_edge_retries_when_adjacency_changes_underneath(store, monkeypatch):
    store.add_edge("ns", "a", "knows", "b")
    real_get_node = store.get_node
    raced = []

    def get_node_then_race(ns, entity_id):
        node = real_get_node(ns, entity_id)
        if not raced:  # a concurrent writer appends after our first read
            raced.append(True)
            store.add_edge("ns", "a", "knows", "c")
        return node

    monkeypatch.setattr(store, "get_node", get_node_then_race)

    assert store.delete_edge("ns", "a", "knows", "b") == 1
    # the concurrent append was not overwritten by a stale list
    assert _edges(store, "ns", "a") == [("knows", "c")]


# ----------------------------------------------------------------- delete_node
def test_delete_node_removes_node_and_inbound_edges(store):
    store.add_edge("ns", "a", "knows", "target")
    store.add_edge("ns", "a", "knows", "b")
    store.add_edge("ns", "b", "owns", "target")
    store.add_edge("ns", "target", "knows", "a")  # outbound: goes with the node
    store.add_edge("ns", "target", "self", "target")

    assert store.delete_node("ns", "target") == 2

    assert store.get_node("ns", "target") is None
    assert _edges(store, "ns", "a") == [("knows", "b")]
    assert _edges(store, "ns", "b") == []
    assert store.list_node_ids("ns") == ["a", "b"]


def test_delete_node_is_scoped_to_its_namespace(store):
    store.add_edge("ns", "a", "knows", "target")
    store.add_edge("other", "a", "knows", "target")

    assert store.delete_node("ns", "target") == 1

    assert store.get_node("other", "target") is not None
    assert _edges(store, "other", "a") == [("knows", "target")]


def test_delete_node_is_idempotent(store):
    store.add_edge("ns", "a", "knows", "target")

    assert store.delete_node("ns", "target") == 1
    assert store.delete_node("ns", "target") == 0
    assert store.delete_node("ns", "never-existed") == 0
    assert store.list_node_ids("ns") == ["a"]


def test_delete_node_escapes_keys(store):
    store.add_edge("ns#1", "a#b", "knows", "c%d")

    assert store.delete_node("ns#1", "c%d") == 1
    assert store.list_node_ids("ns#1") == ["a#b"]
    assert _edges(store, "ns#1", "a#b") == []


def test_non_conditional_errors_are_not_retried(store, monkeypatch):
    store.add_edge("ns", "a", "knows", "b")
    calls = []

    def denied(**kwargs):
        calls.append(kwargs)
        raise ClientError({"Error": {"Code": "AccessDeniedException"}}, "UpdateItem")

    monkeypatch.setattr(store._table, "update_item", denied)

    with pytest.raises(ClientError):
        store.delete_edge("ns", "a", "knows", "b")
    assert len(calls) == 1
