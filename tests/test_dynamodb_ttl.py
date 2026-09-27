import time
from unittest.mock import MagicMock

import pytest

from dynavec.client import Dynavec
from dynavec.config import DynavecConfig
from dynavec.models import Document, SearchResult
from dynavec.provisioning import ensure_table, ensure_ttl
from dynavec.stores.dynamodb import DynamoDBStore, _build_item, check_item_size


def test_document_validation():
    # Valid positive ttl_seconds
    doc = Document(id="d1", text="valid", ttl_seconds=3600)
    assert doc.ttl_seconds == 3600

    # Non-positive values raise ValueError
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        Document(id="d2", text="invalid", ttl_seconds=0)

    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        Document(id="d3", text="invalid", ttl_seconds=-10)


def test_search_result_ttl_to_dict():
    res = SearchResult(id="d1", score=0.9, text="hello", ttl=1750000000)
    assert res.ttl == 1750000000
    d = res.to_dict()
    assert d["ttl"] == 1750000000

    res_no_ttl = SearchResult(id="d2", score=0.8, text="no ttl")
    assert res_no_ttl.ttl is None
    assert "ttl" not in res_no_ttl.to_dict()


def test_build_item_and_check_item_size():
    now = int(time.time())
    item = _build_item("ns", "doc1", "text", {"tag": "test"}, ttl=now + 3600)
    assert item["pk"] == "ns#doc1"
    assert item["ttl"] == now + 3600

    # Custom TTL attribute
    item_custom = _build_item(
        "ns", "doc1", "text", {"tag": "test"}, ttl=now + 3600, ttl_attribute="expire_epoch"
    )
    assert item_custom["expire_epoch"] == now + 3600
    assert "ttl" not in item_custom

    # check_item_size does not raise for valid item with TTL
    check_item_size("ns", "doc1", "text", {"tag": "test"}, ttl=now + 3600)


def test_upsert_with_ttl_on_document_and_dict():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
    )
    db = Dynavec(cfg)
    db._vectors = MagicMock()
    db._docs = MagicMock()

    t_before = int(time.time())
    db.upsert(
        [
            Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0], ttl_seconds=3600),
            {"id": "d2", "vector": [0.0, 1.0, 0.0, 0.0], "ttl_seconds": 7200},
        ],
        namespace="test-ns",
    )
    t_after = int(time.time())

    assert db._docs.put_many.called
    args, _ = db._docs.put_many.call_args
    ns, ddb_payload = args
    assert ns == "test-ns"
    assert len(ddb_payload) == 2

    # Check d1 payload: (id, text, metadata) with _ttl in metadata
    id1, _, meta1 = ddb_payload[0]
    assert id1 == "d1"
    ttl1 = meta1["_ttl"]
    assert t_before + 3600 <= ttl1 <= t_after + 3600

    # Check d2 payload
    id2, _, meta2 = ddb_payload[1]
    assert id2 == "d2"
    ttl2 = meta2["_ttl"]
    assert t_before + 7200 <= ttl2 <= t_after + 7200


def test_upsert_with_default_ttl_seconds_and_override():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
    )
    db = Dynavec(cfg)
    db._vectors = MagicMock()
    db._docs = MagicMock()

    t_before = int(time.time())
    # d1 uses batch default ttl_seconds=1800; d2 overrides with ttl_seconds=9000
    db.upsert(
        [
            Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0]),
            Document(id="d2", vector=[0.0, 1.0, 0.0, 0.0], ttl_seconds=9000),
        ],
        namespace="test-ns",
        ttl_seconds=1800,
    )
    t_after = int(time.time())

    args, _ = db._docs.put_many.call_args
    _, ddb_payload = args
    _, _, meta1 = ddb_payload[0]
    _, _, meta2 = ddb_payload[1]
    ttl1 = meta1["_ttl"]
    ttl2 = meta2["_ttl"]

    assert t_before + 1800 <= ttl1 <= t_after + 1800
    assert t_before + 9000 <= ttl2 <= t_after + 9000


def test_upsert_rejects_negative_ttl_seconds():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
    )
    db = Dynavec(cfg)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        db.upsert([Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0])], ttl_seconds=-5)


def test_namespace_view_upsert_and_update():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
    )
    db = Dynavec(cfg)
    db.upsert = MagicMock()
    db.update = MagicMock()

    ns = db.namespace("prod")
    ns.upsert([Document(id="x", vector=[1.0, 0.0, 0.0, 0.0])], ttl_seconds=500)
    db.upsert.assert_called_once()
    assert db.upsert.call_args[1]["ttl_seconds"] == 500
    assert db.upsert.call_args[1]["namespace"] == "prod"

    ns.update("x", text="new text", ttl_seconds=600)
    db.update.assert_called_once()
    assert db.update.call_args[1]["ttl_seconds"] == 600
    assert db.update.call_args[1]["namespace"] == "prod"


def test_update_ttl_and_preservation():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
    )
    db = Dynavec(cfg)
    db._vectors = MagicMock()
    db._docs = MagicMock()

    # Case 1: Updating document with new ttl_seconds
    db._docs.get_versioned.return_value = {
        "text": "old",
        "metadata": {},
        "version": 1,
        "ttl": 1700000000,
    }
    t_before = int(time.time())
    db.update("doc1", vector=[1.0, 0.0, 0.0, 0.0], ttl_seconds=1200)
    t_after = int(time.time())

    call_meta = db._docs.put_versioned.call_args[0][3]
    assert t_before + 1200 <= call_meta["_ttl"] <= t_after + 1200

    # Case 2: Updating without specifying ttl_seconds preserves existing ttl
    db.update("doc1", vector=[1.0, 0.0, 0.0, 0.0])
    call_meta2 = db._docs.put_versioned.call_args[0][3]
    assert call_meta2["_ttl"] == 1700000000


@pytest.fixture
def moto_store():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )
        session.client("dynamodb").create_table(
            TableName="test_docs",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        config = DynavecConfig(
            vector_bucket="test-bucket",
            index="test-index",
            table="test_docs",
            dimension=4,
            region="us-east-1",
        )
        yield DynamoDBStore(config, boto_session=session), session, config


def test_dynamodb_store_ttl_roundtrip(moto_store):
    store, session, config = moto_store
    exp_time = int(time.time()) + 3600

    # put_many with TTL
    store.put_many("ns", [("d1", "Text 1", {"k": "v"}, exp_time)])

    hydrated = store.get_many("ns", ["d1"])
    assert "d1" in hydrated
    assert hydrated["d1"]["ttl"] == exp_time
    assert hydrated["d1"]["text"] == "Text 1"

    # put_versioned with TTL
    store.put_versioned("ns", "d2", "Text 2", {"k": "v2"}, expected_version=0, ttl=exp_time + 100)
    versioned = store.get_versioned("ns", "d2")
    assert versioned is not None
    assert versioned["ttl"] == exp_time + 100
    assert versioned["version"] == 1


def test_ensure_ttl_and_ensure_table(moto_store):
    _, session, config = moto_store
    ddb = session.client("dynamodb")

    # Initial state: table exists without TTL
    desc = ddb.describe_time_to_live(TableName=config.table)
    assert desc["TimeToLiveDescription"]["TimeToLiveStatus"] == "DISABLED"

    # ensure_ttl enables it
    ensure_ttl(config, session)
    desc_after = ddb.describe_time_to_live(TableName=config.table)
    assert desc_after["TimeToLiveDescription"]["TimeToLiveStatus"] == "ENABLED"
    assert desc_after["TimeToLiveDescription"]["AttributeName"] == "ttl"

    # Idempotent call succeeds
    ensure_ttl(config, session)

    # ensure_table on existing table with TTL enabled succeeds
    ensure_table(config, session)
