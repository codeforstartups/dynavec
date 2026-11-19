"""Unit tests for DynamoDB telemetry events persistence (Issue #127)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import boto3
import pytest
from moto import mock_aws

from dynavec import (
    DynamoDBTelemetryEvent,
    DynamoDBTelemetryRecorder,
    DynamoDBTelemetrySink,
    Dynavec,
    DynavecConfig,
    TelemetryEvent,
    create_telemetry_table,
)
from dynavec.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    """Deterministic dummy embedder for unit tests."""

    def __init__(self, dimension: int = 4) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


def _create_mock_client() -> Any:
    return boto3.client("dynamodb", region_name="us-east-1")


def _create_mock_resource() -> Any:
    return boto3.resource("dynamodb", region_name="us-east-1")


def test_create_telemetry_table_provisioning_and_ttl():
    """Verify table provisioning and TTL activation via create_telemetry_table."""
    with mock_aws():
        client = _create_mock_client()

        # Provision table with PAY_PER_REQUEST billing
        desc = create_telemetry_table(
            client,
            table_name="dynavec-telemetry",
            ttl_attribute="ttl",
            billing_mode="PAY_PER_REQUEST",
        )
        assert desc["Table"]["TableName"] == "dynavec-telemetry"
        assert desc["Table"]["BillingModeSummary"]["BillingMode"] == "PAY_PER_REQUEST"

        # Key schema check
        key_schema = {k["AttributeName"]: k["KeyType"] for k in desc["Table"]["KeySchema"]}
        assert key_schema == {"pk": "HASH", "sk": "RANGE"}

        # TTL activation check
        ttl_desc = client.describe_time_to_live(TableName="dynavec-telemetry")
        assert ttl_desc["TimeToLiveDescription"]["TimeToLiveStatus"] in ("ENABLED", "ENABLING")
        assert ttl_desc["TimeToLiveDescription"]["AttributeName"] == "ttl"

        # Idempotency check: calling again should not fail
        desc2 = create_telemetry_table(client, table_name="dynavec-telemetry")
        assert desc2["Table"]["TableName"] == "dynavec-telemetry"


def test_create_telemetry_table_provisioned_mode():
    """Verify table provisioning with PROVISIONED billing mode."""
    with mock_aws():
        client = _create_mock_client()
        desc = create_telemetry_table(
            client,
            table_name="telemetry-provisioned",
            billing_mode="PROVISIONED",
            read_capacity_units=10,
            write_capacity_units=10,
        )
        tp = desc["Table"]["ProvisionedThroughput"]
        assert tp["ReadCapacityUnits"] == 10
        assert tp["WriteCapacityUnits"] == 10


def test_record_buffering_and_batch_writes():
    """Verify buffering up to buffer_size and explicit flush using batch writes."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        sink = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            buffer_size=4,
            async_flush=False,
        )

        # Record 3 events — should remain buffered
        for i in range(3):
            sink.record(
                sink.new_event(
                    "search",
                    namespace="test_ns",
                    latency_ms=10.0 + i,
                )
            )

        events_in_db = sink.query_by_namespace("test_ns")
        assert len(events_in_db) == 0

        # 4th event triggers flush on buffer limit
        sink.record(sink.new_event("search", namespace="test_ns", latency_ms=13.0))
        events_in_db = sink.query_by_namespace("test_ns")
        assert len(events_in_db) == 4

        # Record 2 more and call explicit flush
        sink.record(sink.new_event("search", namespace="test_ns", latency_ms=14.0))
        sink.record(sink.new_event("search", namespace="test_ns", latency_ms=15.0))
        sink.flush()

        events_in_db = sink.query_by_namespace("test_ns")
        assert len(events_in_db) == 6
        sink.close()


def test_batch_write_chunking_exceeding_25_items():
    """Verify that batches larger than 25 items are cleanly chunked and written."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        sink = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            buffer_size=100,
            async_flush=False,
        )

        for i in range(35):
            sink.record(
                sink.new_event("search", namespace="chunk_ns", latency_ms=float(i))
            )
        sink.flush()

        events = sink.query_by_namespace("chunk_ns", limit=100)
        assert len(events) == 35
        sink.close()


def test_non_blocking_async_flush():
    """Verify non-blocking asynchronous flush and graceful shutdown."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        sink = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            buffer_size=3,
            async_flush=True,
        )

        # Write 3 events to trigger async flush
        for i in range(3):
            sink.record(sink.new_event("search", namespace="async_ns", latency_ms=5.0 + i))

        # close() flushes and shuts down thread pool
        sink.close()

        events = sink.query_by_namespace("async_ns")
        assert len(events) == 3


def test_event_serialization_and_ttl_calculation():
    """Verify schema attributes, stage_latencies serialization, and TTL calculation."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        sink = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            ttl_seconds=7 * 86400,
            async_flush=False,
        )

        now = time.time()
        # 1. Event with explicit TTL
        ev_explicit = TelemetryEvent(
            id="q_exp_1",
            ts=now,
            op="search",
            namespace="ser_ns",
            top_k=10,
            latency_ms=12.5,
            stage_latencies={"embed": 3.2, "vector_search": 8.1, "hydration": 1.2},
            result_ids=["doc_1", "doc_2", "doc_3"],
            cache_hit=False,
            ttl=1999999999,
        )
        sink.record(ev_explicit)

        # 2. Event with default TTL calculation (1 second later for deterministic ordering)
        ev_auto = TelemetryEvent(
            id="q_auto_2",
            ts=now + 1.0,
            op="search",
            namespace="ser_ns",
            top_k=5,
            latency_ms=7.0,
            stage_latencies={"vector_search": 5.0},
            result_ids=["doc_4"],
            cache_hit=True,
        )
        sink.record(ev_auto)
        sink.flush()

        results = sink.query_by_namespace("ser_ns", ascending=True)
        assert len(results) == 2

        # Verify explicit TTL
        item1 = results[0]
        assert item1.query_id == "q_exp_1"
        assert item1.namespace == "ser_ns"
        assert item1.top_k == 10
        assert item1.stage_latencies == {"embed": 3.2, "vector_search": 8.1, "hydration": 1.2}
        assert item1.result_ids == ["doc_1", "doc_2", "doc_3"]
        assert item1.cache_hit is False
        assert item1.ttl == 1999999999

        # Verify auto calculated TTL
        item2 = results[1]
        assert item2.query_id == "q_auto_2"
        assert item2.ttl is not None
        assert abs(item2.ttl - int(now + 7 * 86400)) <= 2
        assert item2.cache_hit is True
        assert item2.result_ids == ["doc_4"]
        sink.close()


def test_query_by_namespace_and_time_filtering():
    """Verify querying events by namespace with chronological ordering and time range."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        sink = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            async_flush=False,
        )

        t1 = datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 19, 11, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)

        # Namespace A
        sink.record(
            DynamoDBTelemetryEvent(
                query_id="qa_1",
                namespace="ns_a",
                timestamp=t1.isoformat(),
                latency_ms=10.0,
            )
        )
        sink.record(
            DynamoDBTelemetryEvent(
                query_id="qa_2",
                namespace="ns_a",
                timestamp=t2.isoformat(),
                latency_ms=20.0,
            )
        )
        sink.record(
            DynamoDBTelemetryEvent(
                query_id="qa_3",
                namespace="ns_a",
                timestamp=t3.isoformat(),
                latency_ms=30.0,
            )
        )

        # Namespace B
        sink.record(
            DynamoDBTelemetryEvent(
                query_id="qb_1",
                namespace="ns_b",
                timestamp=t1.isoformat(),
                latency_ms=50.0,
            )
        )
        sink.flush()

        # Query ns_a newest first (default descending)
        ns_a_desc = sink.query_by_namespace("ns_a", ascending=False)
        assert len(ns_a_desc) == 3
        assert [e.query_id for e in ns_a_desc] == ["qa_3", "qa_2", "qa_1"]

        # Query ns_a oldest first (ascending)
        ns_a_asc = sink.query_by_namespace("ns_a", ascending=True)
        assert [e.query_id for e in ns_a_asc] == ["qa_1", "qa_2", "qa_3"]

        # Query with since filtering (t2 onwards)
        ns_a_since = sink.query_by_namespace("ns_a", since=t2.isoformat(), ascending=True)
        assert len(ns_a_since) == 2
        assert [e.query_id for e in ns_a_since] == ["qa_2", "qa_3"]

        # Query ns_b isolation
        ns_b_events = sink.query_by_namespace("ns_b")
        assert len(ns_b_events) == 1
        assert ns_b_events[0].query_id == "qb_1"

        # Query empty namespace
        assert sink.query_by_namespace("nonexistent") == []
        sink.close()


def test_raw_query_omitted_by_default():
    """Verify that raw query text and prompts are never stored unless capture_text=True."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        # 1. Default capture_text=False
        sink_secure = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            capture_text=False,
            async_flush=False,
        )
        sink_secure.record(
            TelemetryEvent(
                id="sec_1",
                ts=time.time(),
                op="search",
                namespace="privacy",
                query_preview="sensitive secret user prompt or SSN",
            )
        )
        sink_secure.flush()

        # Verify through low-level DynamoDB query
        resp = client.query(
            TableName="dynavec-telemetry",
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "privacy#events"}},
        )
        assert resp["Count"] == 1
        raw_item = resp["Items"][0]
        assert "query_preview" not in raw_item
        assert "prompt" not in raw_item
        sink_secure.close()

        # 2. Opt-in capture_text=True
        sink_optin = DynamoDBTelemetrySink(
            client=client,
            table_name="dynavec-telemetry",
            capture_text=True,
            async_flush=False,
        )
        sink_optin.record(
            TelemetryEvent(
                id="opt_1",
                ts=time.time(),
                op="search",
                namespace="privacy_optin",
                query_preview="allowed public query",
            )
        )
        sink_optin.flush()

        resp2 = client.query(
            TableName="dynavec-telemetry",
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "privacy_optin#events"}},
        )
        assert resp2["Count"] == 1
        assert "query_preview" in resp2["Items"][0]
        assert resp2["Items"][0]["query_preview"]["S"] == "allowed public query"
        sink_optin.close()


def test_dynamodb_telemetry_recorder_hybrid():
    """Verify DynamoDBTelemetryRecorder provides both in-memory ring buffer and persistent store."""
    with mock_aws():
        client = _create_mock_client()
        create_telemetry_table(client, table_name="dynavec-telemetry")

        with DynamoDBTelemetryRecorder(
            client=client,
            table_name="dynavec-telemetry",
            max_events=10,
            buffer_size=5,
            async_flush=False,
        ) as rec:
            for i in range(5):
                rec.record(
                    rec.new_event(
                        "search",
                        namespace="hybrid_ns",
                        latency_ms=float(i * 10),
                    )
                )

            # In-memory check
            mem_events = rec.events()
            assert len(mem_events) == 5

            # DynamoDB query check (triggered by buffer_size=5)
            db_events = rec.query_dynamodb("hybrid_ns")
            assert len(db_events) == 5


def test_table_resource_support():
    """Verify DynamoDBTelemetrySink works with boto3 Table resource."""
    with mock_aws():
        res = _create_mock_resource()
        table = res.create_table(
            TableName="resource-telemetry",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        sink = DynamoDBTelemetrySink(table=table, async_flush=False)
        sink.record(
            TelemetryEvent(
                id="res_q1",
                ts=time.time(),
                op="search",
                namespace="res_ns",
                latency_ms=15.0,
            )
        )
        sink.flush()

        events = sink.query_by_namespace("res_ns")
        assert len(events) == 1
        assert events[0].query_id == "res_q1"
        sink.close()


class FakeS3Store:
    """Mock for S3VectorsStore for end-to-end client integration testing."""

    def __init__(self, config: Any, boto_session: Any = None) -> None:
        self.config = config
        self._store: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def put_vectors(self, vectors: Any) -> None:
        for key, vec, meta in vectors:
            self._store[key] = (list(vec), dict(meta))

    def query(
        self,
        query_vector: Any,
        top_k: int,
        filter: Any = None,
        return_metadata: bool = True,
        return_distance: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [{"key": key, "distance": 0.05} for key in list(self._store.keys())[:top_k]]


def test_dynavec_client_integration(monkeypatch: pytest.MonkeyPatch):
    """Verify Dynavec client records telemetry to DynamoDB when telemetry_dynamodb is enabled."""
    monkeypatch.setattr("dynavec.client.S3VectorsStore", FakeS3Store)

    with mock_aws():
        ddb_client = _create_mock_client()
        create_telemetry_table(ddb_client, table_name="dynavec-telemetry")

        cfg = DynavecConfig(
            vector_bucket="test-vectors",
            index="docs",
            table="dynavec_docs",
            dimension=4,
            region="us-east-1",
            auto_provision=False,
            telemetry_dynamodb=True,
            telemetry_table="dynavec-telemetry",
        )

        # Create main documents table
        ddb_client.create_table(
            TableName="dynavec_docs",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        db = Dynavec(cfg, embedder=FakeEmbedder(dimension=4))
        # Upsert a document
        db.upsert([{"id": "doc_1", "text": "hello test"}])

        # Search with explain=True to collect stage latencies
        res = db.search("test query", top_k=2, explain=True)
        assert len(res.results) >= 1

        # Flush telemetry records
        db.flush_telemetry()

        # Check DynamoDB telemetry table
        sink = DynamoDBTelemetrySink(client=ddb_client, table_name="dynavec-telemetry")
        events = sink.query_by_namespace("default")
        assert len(events) >= 1
        last_event = events[0]
        assert last_event.op == "search"
        assert last_event.top_k == 2
        assert isinstance(last_event.stage_latencies, dict)
        assert last_event.result_ids == ["doc_1"]
        assert last_event.query_preview is None  # privacy preserved

        db.close()
        sink.close()
