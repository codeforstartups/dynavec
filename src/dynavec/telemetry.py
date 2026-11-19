"""Telemetry capture for the observability dashboard.

This records **real** events from actual dynavec operations — every ``search``
is timed and logged with its namespace, latency, result count, cache outcome,
and score stats. There is no simulated traffic: if the dashboard shows numbers,
they came from real calls into the client.

Design
------
* :class:`TelemetryEvent` — one recorded operation (a "trace").
* :class:`TelemetryRecorder` — thread-safe ring buffer + aggregation. Attach it
  to a client (``Dynavec(..., telemetry=recorder)``) and it fills as you query.
* :func:`aggregate` — percentiles, QPM, cache-hit-rate, op mix, error rate,
  per-minute histogram — computed from the real events.

Zero third-party deps; safe to import with just the base install.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """A single recorded operation."""

    id: str
    ts: float  # epoch seconds (start)
    op: str  # "search" | "upsert" | "graph_search" | ...
    namespace: str = "default"
    latency_ms: float = 0.0
    n_results: int = 0
    top_k: int | None = None
    cache_hit: bool | None = None  # None = no cache configured
    filtered: bool = False
    rescore: str | None = None
    rerank: str | None = None
    score_top: float | None = None
    score_mean: float | None = None
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    query_preview: str | None = None  # only set when capture_text=True
    eval_faithfulness: float | None = None  # 0.0-1.0 when LLM judge ran
    eval_relevance: float | None = None  # 0.0-1.0 when LLM judge ran
    eval_recall: float | None = None  # 0.0-1.0 recall@k score
    eval_mrr: float | None = None  # 0.0-1.0 mean reciprocal rank
    eval_ndcg: float | None = None  # 0.0-1.0 ndcg@k score
    query_id: str | None = None
    stage_latencies: dict[str, float] = field(default_factory=dict)
    result_ids: list[str] = field(default_factory=list)
    timestamp: str | None = None
    ttl: int | None = None

    def __post_init__(self) -> None:
        if self.query_id is None:
            self.query_id = self.id
        if self.timestamp is None:
            self.timestamp = datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamoDBTelemetryEvent:
    """Telemetry event model for persistent DynamoDB storage."""

    query_id: str
    namespace: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    top_k: int | None = None
    stage_latencies: dict[str, float] = field(default_factory=dict)
    result_ids: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    ttl: int | None = None
    latency_ms: float = 0.0
    status: str = "ok"
    error: str | None = None
    query_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    op: str = "search"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dynamo_item(
        self,
        *,
        ttl_seconds: int | None = 30 * 86400,
        capture_text: bool = False,
        ttl_attribute: str = "ttl",
    ) -> dict[str, Any]:
        from .logging import redact_secrets
        from .stores.dynamodb import _to_dynamo

        ttl_val = self.ttl
        if ttl_val is None and ttl_seconds is not None and ttl_seconds > 0:
            ttl_val = int(time.time() + ttl_seconds)

        item: dict[str, Any] = {
            "pk": f"{self.namespace}#events",
            "sk": f"{self.timestamp}#{self.query_id}",
            "query_id": self.query_id,
            "namespace": self.namespace,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "op": self.op,
            "stage_latencies": dict(self.stage_latencies),
            "result_ids": list(self.result_ids),
        }
        if self.top_k is not None:
            item["top_k"] = self.top_k
        if self.cache_hit is not None:
            item["cache_hit"] = self.cache_hit
        if ttl_val is not None:
            item[ttl_attribute] = int(ttl_val)
        if self.error is not None:
            item["error"] = str(redact_secrets(self.error))
        # Never store raw prompts or PII unless explicitly enabled
        if capture_text and self.query_preview:
            item["query_preview"] = str(self.query_preview)[:80]
        if self.metadata:
            item["metadata"] = redact_secrets(self.metadata)

        return cast(dict[str, Any], _to_dynamo(item))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamoDBTelemetryEvent:
        query_id = data.get("query_id")
        if not query_id and "sk" in data:
            parts = str(data["sk"]).split("#", 1)
            if len(parts) == 2:
                query_id = parts[1]
        if not query_id:
            query_id = data.get("id") or uuid.uuid4().hex[:12]

        timestamp = data.get("timestamp")
        if not timestamp and "sk" in data:
            parts = str(data["sk"]).split("#", 1)
            timestamp = parts[0]
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        namespace = data.get("namespace")
        if not namespace and "pk" in data:
            pk_str = str(data["pk"])
            if "#" in pk_str:
                namespace = pk_str.split("#")[0]
        if not namespace:
            namespace = "default"

        top_k = data.get("top_k")
        if top_k is not None:
            top_k = int(top_k)

        ttl = data.get("ttl")
        if ttl is not None:
            ttl = int(ttl)

        return cls(
            query_id=str(query_id),
            namespace=str(namespace),
            timestamp=str(timestamp),
            top_k=top_k,
            stage_latencies=dict(data.get("stage_latencies") or {}),
            result_ids=list(data.get("result_ids") or []),
            cache_hit=data.get("cache_hit"),
            ttl=ttl,
            latency_ms=float(data.get("latency_ms", 0.0)),
            status=str(data.get("status", "ok")),
            error=data.get("error"),
            query_preview=data.get("query_preview"),
            metadata=dict(data.get("metadata") or {}),
            op=str(data.get("op", "search")),
        )


class DynamoDBTelemetrySink:
    """DynamoDB sink for persistent telemetry events.

    Buffers events in memory and writes them in batches via batch_write_item
    to avoid blocking query latency. Writes can be triggered on buffer capacity,
    or explicit .flush() / .close().
    """

    def __init__(
        self,
        table: Any = None,
        *,
        table_name: str = "dynavec-telemetry",
        client: Any = None,
        boto_session: Any = None,
        region_name: str | None = None,
        namespace: str = "default",
        buffer_size: int = 25,
        async_flush: bool = True,
        ttl_seconds: int = 30 * 86400,
        ttl_attribute: str = "ttl",
        capture_text: bool = False,
    ) -> None:
        self.namespace = namespace
        self.buffer_size = max(1, buffer_size)
        self.async_flush = async_flush
        self.ttl_seconds = ttl_seconds
        self.ttl_attribute = ttl_attribute
        self.capture_text = capture_text
        self.table_name = table_name

        self.table: Any = None
        self.client: Any = None

        if table is not None:
            if isinstance(table, str):
                self.table_name = table
            elif hasattr(table, "batch_writer"):
                self.table = table
                self.table_name = getattr(table, "name", getattr(table, "table_name", table_name))
                if client is None and hasattr(table, "meta") and hasattr(table.meta, "client"):
                    self.client = table.meta.client
            elif hasattr(table, "batch_write_item"):
                self.client = table

        if client is not None:
            if hasattr(client, "meta") and hasattr(client.meta, "client"):
                self.client = client.meta.client
            else:
                self.client = client

        if self.table is None and self.client is None:
            import boto3

            session = boto_session or boto3.Session()
            client_kwargs: dict[str, Any] = {}
            if region_name:
                client_kwargs["region_name"] = region_name
            self.client = session.client("dynamodb", **client_kwargs)

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dynavec-telemetry")
        self._active_futures: list[Future[None]] = []
        self._closed = False

    def new_event(self, op: str = "search", **kw: Any) -> TelemetryEvent:
        event_id = kw.pop("id", None) or uuid.uuid4().hex[:12]
        query_id = kw.pop("query_id", None) or event_id
        ts = kw.pop("ts", None) or time.time()
        timestamp = (
            kw.pop("timestamp", None)
            or datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        )
        return TelemetryEvent(
            id=event_id,
            query_id=query_id,
            ts=ts,
            timestamp=timestamp,
            op=op,
            **kw,
        )

    def _serialize_event(
        self,
        event: TelemetryEvent | DynamoDBTelemetryEvent | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(event, DynamoDBTelemetryEvent):
            return event.to_dynamo_item(
                ttl_seconds=self.ttl_seconds,
                capture_text=self.capture_text,
                ttl_attribute=self.ttl_attribute,
            )
        if isinstance(event, TelemetryEvent):
            q_id = getattr(event, "query_id", None) or event.id
            ts = getattr(event, "ts", time.time())
            iso_ts = (
                getattr(event, "timestamp", None)
                or datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            )
            db_event = DynamoDBTelemetryEvent(
                query_id=q_id,
                namespace=getattr(event, "namespace", self.namespace) or self.namespace,
                timestamp=iso_ts,
                top_k=getattr(event, "top_k", None),
                stage_latencies=dict(getattr(event, "stage_latencies", None) or {}),
                result_ids=list(getattr(event, "result_ids", None) or []),
                cache_hit=getattr(event, "cache_hit", None),
                ttl=getattr(event, "ttl", None),
                latency_ms=getattr(event, "latency_ms", 0.0),
                status=getattr(event, "status", "ok"),
                error=getattr(event, "error", None),
                query_preview=getattr(event, "query_preview", None),
                op=getattr(event, "op", "search"),
            )
            return db_event.to_dynamo_item(
                ttl_seconds=self.ttl_seconds,
                capture_text=self.capture_text,
                ttl_attribute=self.ttl_attribute,
            )
        if isinstance(event, dict):
            data = dict(event)
            if "namespace" not in data:
                data["namespace"] = self.namespace
            db_event = DynamoDBTelemetryEvent.from_dict(data)
            return db_event.to_dynamo_item(
                ttl_seconds=self.ttl_seconds,
                capture_text=self.capture_text,
                ttl_attribute=self.ttl_attribute,
            )
        raise TypeError(f"Unsupported event type: {type(event)}")

    def record(
        self,
        event: TelemetryEvent | DynamoDBTelemetryEvent | dict[str, Any],
    ) -> None:
        """Buffer a telemetry event and flush if buffer limit is reached."""
        if self._closed:
            raise RuntimeError("Cannot record telemetry on a closed DynamoDBTelemetrySink")

        item = self._serialize_event(event)

        items_to_flush: list[dict[str, Any]] | None = None
        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self.buffer_size:
                items_to_flush = list(self._buffer)
                self._buffer.clear()

        if items_to_flush:
            self._dispatch_write(items_to_flush)

    def _dispatch_write(self, items: list[dict[str, Any]]) -> None:
        if self.async_flush:
            with self._lock:
                self._active_futures = [f for f in self._active_futures if not f.done()]
                future = self._executor.submit(self._write_batch, items)
                self._active_futures.append(future)
        else:
            self._write_batch(items)

    def _write_batch(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return

        from .stores.dynamodb import _to_dynamo

        for i in range(0, len(items), 25):
            chunk = items[i : i + 25]
            if self.table is not None and hasattr(self.table, "batch_writer"):
                try:
                    with self.table.batch_writer() as batch:
                        for it in chunk:
                            batch.put_item(Item=_to_dynamo(it))
                except Exception as exc:
                    logger.warning(
                        "Failed to batch write telemetry items to table %s: %s",
                        self.table_name,
                        exc,
                    )
            elif self.client is not None:
                from boto3.dynamodb.types import TypeSerializer

                serializer = TypeSerializer()
                request_items = {
                    self.table_name: [
                        {
                            "PutRequest": {
                                "Item": {
                                    k: serializer.serialize(v)
                                    for k, v in _to_dynamo(it).items()
                                }
                            }
                        }
                        for it in chunk
                    ]
                }
                retries = 3
                backoff = 0.05
                unprocessed = request_items
                while unprocessed and retries > 0:
                    try:
                        resp = self.client.batch_write_item(RequestItems=unprocessed)
                        unprocessed = resp.get("UnprocessedItems", {})
                        if unprocessed:
                            time.sleep(backoff)
                            backoff *= 2
                            retries -= 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to batch_write_item telemetry to %s: %s",
                            self.table_name,
                            exc,
                        )
                        break

    def flush(self) -> None:
        """Flush all buffered events and wait for in-flight asynchronous writes."""
        with self._lock:
            items_to_flush = list(self._buffer)
            self._buffer.clear()
            futures = list(self._active_futures)

        for f in futures:
            try:
                f.result()
            except Exception as exc:
                logger.warning("Error in background telemetry flush: %s", exc)

        with self._lock:
            self._active_futures = [f for f in self._active_futures if not f.done()]

        if items_to_flush:
            self._write_batch(items_to_flush)

    def query_by_namespace(
        self,
        namespace: str = "default",
        limit: int = 100,
        ascending: bool = False,
        since: str | float | None = None,
        until: str | float | None = None,
    ) -> list[DynamoDBTelemetryEvent]:
        """Query telemetry events chronologically by namespace partition key."""
        from .stores.dynamodb import _from_dynamo

        if isinstance(since, (int, float)):
            since = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
        if isinstance(until, (int, float)):
            until = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()

        if self.table is not None and hasattr(self.table, "query"):
            from boto3.dynamodb.conditions import Key

            key_cond = Key("pk").eq(f"{namespace}#events")
            if since is not None and until is not None:
                key_cond = key_cond & Key("sk").between(str(since), str(until) + "~")
            elif since is not None:
                key_cond = key_cond & Key("sk").gte(str(since))
            elif until is not None:
                key_cond = key_cond & Key("sk").lte(str(until) + "~")

            resp = self.table.query(
                KeyConditionExpression=key_cond,
                ScanIndexForward=ascending,
                Limit=limit,
            )
            raw_items = [_from_dynamo(it) for it in resp.get("Items", [])]
            return [DynamoDBTelemetryEvent.from_dict(it) for it in raw_items]

        if self.client is not None:
            from boto3.dynamodb.types import TypeDeserializer

            key_conditions = ["#pk = :pk"]
            attr_names: dict[str, str] = {"#pk": "pk"}
            attr_values: dict[str, Any] = {":pk": {"S": f"{namespace}#events"}}

            if since is not None and until is not None:
                key_conditions.append("#sk BETWEEN :since AND :until")
                attr_names["#sk"] = "sk"
                attr_values[":since"] = {"S": str(since)}
                attr_values[":until"] = {"S": str(until) + "~"}
            elif since is not None:
                key_conditions.append("#sk >= :since")
                attr_names["#sk"] = "sk"
                attr_values[":since"] = {"S": str(since)}
            elif until is not None:
                key_conditions.append("#sk <= :until")
                attr_names["#sk"] = "sk"
                attr_values[":until"] = {"S": str(until) + "~"}

            resp = self.client.query(
                TableName=self.table_name,
                KeyConditionExpression=" AND ".join(key_conditions),
                ExpressionAttributeNames=attr_names,
                ExpressionAttributeValues=attr_values,
                ScanIndexForward=ascending,
                Limit=limit,
            )
            deserializer = TypeDeserializer()
            raw_items = [
                _from_dynamo({k: deserializer.deserialize(v) for k, v in raw_it.items()})
                for raw_it in resp.get("Items", [])
            ]
            return [DynamoDBTelemetryEvent.from_dict(it) for it in raw_items]

        return []

    def close(self) -> None:
        """Flush remaining events and shut down background executor."""
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> DynamoDBTelemetrySink:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class TelemetryRecorder:
    """Thread-safe, bounded, in-process store of :class:`TelemetryEvent`.

    Parameters
    ----------
    max_events:
        Ring-buffer capacity (older events drop off).
    capture_text:
        If True, store a short preview of the query text. Off by default so no
        raw content is retained unless the operator opts in.
    """

    def __init__(self, max_events: int = 10_000, capture_text: bool = False) -> None:
        self._events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self.capture_text = capture_text

    # ------------------------------------------------------------------ record
    def record(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)

    def new_event(self, op: str, **kw: Any) -> TelemetryEvent:
        event_id = kw.pop("id", None) or uuid.uuid4().hex[:12]
        query_id = kw.pop("query_id", None) or event_id
        ts = kw.pop("ts", None) or time.time()
        timestamp = (
            kw.pop("timestamp", None)
            or datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        )
        return TelemetryEvent(
            id=event_id,
            query_id=query_id,
            ts=ts,
            timestamp=timestamp,
            op=op,
            **kw,
        )

    # ------------------------------------------------------------------- reads
    def events(
        self,
        limit: int = 200,
        op: str | None = None,
        namespace: str | None = None,
        status: str | None = None,
        since: float | None = None,
    ) -> list[TelemetryEvent]:
        with self._lock:
            items = list(self._events)
        out = []
        for e in reversed(items):  # newest first
            if op and e.op != op:
                continue
            if namespace and e.namespace != namespace:
                continue
            if status and e.status != status:
                continue
            if since and e.ts < since:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out

    def get(self, event_id: str) -> TelemetryEvent | None:
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    return e
        return None

    def snapshot(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class DynamoDBTelemetryRecorder(TelemetryRecorder):
    """Hybrid telemetry recorder: in-memory ring buffer + persistent DynamoDB sink."""

    def __init__(
        self,
        table: Any = None,
        *,
        table_name: str = "dynavec-telemetry",
        client: Any = None,
        boto_session: Any = None,
        region_name: str | None = None,
        namespace: str = "default",
        max_events: int = 10_000,
        capture_text: bool = False,
        buffer_size: int = 25,
        async_flush: bool = True,
        ttl_seconds: int = 30 * 86400,
        ttl_attribute: str = "ttl",
    ) -> None:
        super().__init__(max_events=max_events, capture_text=capture_text)
        self.sink = DynamoDBTelemetrySink(
            table=table,
            table_name=table_name,
            client=client,
            boto_session=boto_session,
            region_name=region_name,
            namespace=namespace,
            buffer_size=buffer_size,
            async_flush=async_flush,
            ttl_seconds=ttl_seconds,
            ttl_attribute=ttl_attribute,
            capture_text=capture_text,
        )

    def record(
        self,
        event: TelemetryEvent | DynamoDBTelemetryEvent | dict[str, Any],
    ) -> None:
        if isinstance(event, TelemetryEvent):
            super().record(event)
        self.sink.record(event)

    def flush(self) -> None:
        self.sink.flush()

    def close(self) -> None:
        self.sink.close()

    def query_dynamodb(
        self,
        namespace: str = "default",
        limit: int = 100,
        ascending: bool = False,
        since: str | float | None = None,
        until: str | float | None = None,
    ) -> list[DynamoDBTelemetryEvent]:
        return self.sink.query_by_namespace(
            namespace=namespace,
            limit=limit,
            ascending=ascending,
            since=since,
            until=until,
        )

    def query_by_namespace(
        self,
        namespace: str = "default",
        limit: int = 100,
        ascending: bool = False,
        since: str | float | None = None,
        until: str | float | None = None,
    ) -> list[DynamoDBTelemetryEvent]:
        return self.sink.query_by_namespace(
            namespace=namespace,
            limit=limit,
            ascending=ascending,
            since=since,
            until=until,
        )

    def __enter__(self) -> DynamoDBTelemetryRecorder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate(
    events: list[TelemetryEvent],
    window_seconds: int = 3600,
    now: float | None = None,
    buckets: int = 30,
) -> dict[str, Any]:
    """Summarize events over the trailing ``window_seconds`` into dashboard stats."""
    now = now if now is not None else time.time()
    start = now - window_seconds
    win = [e for e in events if e.ts >= start]

    latencies = sorted(e.latency_ms for e in win if e.status == "ok")
    cache_scoped = [e for e in win if e.cache_hit is not None]
    hits = sum(1 for e in cache_scoped if e.cache_hit)
    errors = sum(1 for e in win if e.status == "error")

    faith_scores = [e.eval_faithfulness for e in win if e.eval_faithfulness is not None]
    rel_scores = [e.eval_relevance for e in win if e.eval_relevance is not None]
    recall_scores = [e.eval_recall for e in win if e.eval_recall is not None]
    mrr_scores = [e.eval_mrr for e in win if e.eval_mrr is not None]
    ndcg_scores = [e.eval_ndcg for e in win if e.eval_ndcg is not None]

    op_mix: dict[str, int] = {}
    ns_mix: dict[str, int] = {}
    for e in win:
        op_mix[e.op] = op_mix.get(e.op, 0) + 1
        ns_mix[e.namespace] = ns_mix.get(e.namespace, 0) + 1

    # per-bucket count histogram across the window
    bucket_width = max(1.0, window_seconds / buckets)
    hist = [0] * buckets
    for e in win:
        idx = int((e.ts - start) / bucket_width)
        if 0 <= idx < buckets:
            hist[idx] += 1

    minutes = max(1e-9, window_seconds / 60.0)
    return {
        "total": len(win),
        "qpm": round(len(win) / minutes, 2),
        "p50": round(_percentile(latencies, 50), 2),
        "p95": round(_percentile(latencies, 95), 2),
        "p99": round(_percentile(latencies, 99), 2),
        "cache_hit_rate": round(100.0 * hits / len(cache_scoped), 1) if cache_scoped else None,
        "cache_hits": hits,
        "cache_total": len(cache_scoped),
        "error_rate": round(100.0 * errors / len(win), 2) if win else 0.0,
        "avg_results": round(sum(e.n_results for e in win) / len(win), 1) if win else 0.0,
        "eval_faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4)
        if faith_scores
        else None,
        "eval_relevance_mean": round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
        "eval_recall_mean": round(sum(recall_scores) / len(recall_scores), 4)
        if recall_scores
        else None,
        "eval_mrr_mean": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None,
        "eval_ndcg_mean": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else None,
        "eval_count": max(len(faith_scores), len(rel_scores), len(recall_scores)),
        "op_mix": op_mix,
        "namespaces": ns_mix,
        "histogram": hist,
        "bucket_width_s": bucket_width,
        "window_start": start,
        "window_seconds": window_seconds,
    }


def aggregate_eval(
    events: list[TelemetryEvent],
    window_seconds: int = 86400,
    now: float | None = None,
) -> dict[str, Any]:
    """Summarize evaluation scores across recorded telemetry events."""
    now = now if now is not None else time.time()
    start = now - window_seconds
    win = [e for e in events if e.ts >= start]

    eval_events = [
        e
        for e in win
        if e.eval_faithfulness is not None
        or e.eval_relevance is not None
        or e.eval_recall is not None
        or e.eval_mrr is not None
        or e.eval_ndcg is not None
    ]
    faith_scores = [e.eval_faithfulness for e in eval_events if e.eval_faithfulness is not None]
    rel_scores = [e.eval_relevance for e in eval_events if e.eval_relevance is not None]
    recall_scores = [e.eval_recall for e in eval_events if e.eval_recall is not None]
    mrr_scores = [e.eval_mrr for e in eval_events if e.eval_mrr is not None]
    ndcg_scores = [e.eval_ndcg for e in eval_events if e.eval_ndcg is not None]

    pass_threshold = 0.7
    passed = [
        e
        for e in eval_events
        if (e.eval_faithfulness is None or e.eval_faithfulness >= pass_threshold)
        and (e.eval_relevance is None or e.eval_relevance >= pass_threshold)
        and (e.eval_recall is None or e.eval_recall >= pass_threshold)
    ]

    return {
        "total_evals": len(eval_events),
        "mean_faithfulness": round(sum(faith_scores) / len(faith_scores), 4)
        if faith_scores
        else None,
        "mean_relevance": round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else None,
        "mean_recall": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else None,
        "mean_mrr": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None,
        "mean_ndcg": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else None,
        "pass_rate": round(len(passed) / len(eval_events), 4) if eval_events else None,
        "pass_threshold": pass_threshold,
        "window_seconds": window_seconds,
    }
