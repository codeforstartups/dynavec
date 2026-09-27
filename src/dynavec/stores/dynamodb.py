"""DynamoDB document / metadata store.

Role in the hybrid design:
  * canonical store for the full source text + rich metadata
  * single-digit-ms hydration of documents by id after S3 Vectors returns keys
  * partition-friendly key design (``pk = "{namespace}#{id}"``) that spreads load
    evenly and makes ``BatchGetItem`` hydration O(1) per document

Numbers are stored natively (float -> Decimal) so future GSIs / access patterns
can filter on metadata fields directly in DynamoDB.
"""

from __future__ import annotations

import gzip
import logging
import time
from decimal import Decimal
from typing import Any

from ..config import DynavecConfig
from ..exceptions import ConflictError, ItemTooLargeError
from ..logging import log_store_event
from ..utils import KEY_SEPARATOR, encode_key_component, retry

Metadata = dict[str, Any]

_BATCH_GET_LIMIT = 100

# DynamoDB's hard per-item limit, attribute names included.
MAX_ITEM_BYTES = 400 * 1024

# Optimistic-concurrency counter written by ``put_versioned``. Items written by
# ``put_many`` (plain upserts) carry no version and read back as version 0.
VERSION_ATTR = "version"


def _to_dynamo(obj: Any) -> Any:
    """Recursively convert Python floats to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dynamo(v) for v in obj]
    return obj


def _from_dynamo(obj: Any) -> Any:
    """Recursively convert Decimals back to int/float for callers."""
    if isinstance(obj, Decimal):
        i = int(obj)
        return i if i == obj else float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


def _pk(namespace: str, doc_id: str) -> str:
    return f"{encode_key_component(namespace)}{KEY_SEPARATOR}{encode_key_component(doc_id)}"


def _build_item(
    namespace: str,
    doc_id: str,
    text: str | None,
    metadata: Metadata,
    gzip_threshold_bytes: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "pk": _pk(namespace, doc_id),
        "ns": namespace,
        "id": doc_id,
        "metadata": _to_dynamo(metadata or {}),
    }
    if text is not None:
        text_bytes = text.encode("utf-8")
        if gzip_threshold_bytes is not None and len(text_bytes) >= gzip_threshold_bytes:
            item["text_gzip"] = gzip.compress(text_bytes)
        else:
            item["text"] = text
    return item


def _value_size(value: Any) -> int:
    """Approximate stored size of one attribute value, per DynamoDB's sizing rules."""
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (int, Decimal)):
        # 1 byte per two significant digits, plus 1
        digits = len(Decimal(value).normalize().as_tuple().digits)
        return (digits + 1) // 2 + 1
    if isinstance(value, dict):
        # 3 bytes overhead + 1 per element, plus each key name and value
        return 3 + sum(1 + len(str(k).encode("utf-8")) + _value_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 3 + sum(1 + _value_size(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return sum(_value_size(v) for v in value)
    return len(str(value).encode("utf-8"))


def item_size_bytes(item: dict[str, Any]) -> int:
    """Approximate DynamoDB size of ``item``: attribute name bytes plus value sizes."""
    return sum(len(name.encode("utf-8")) + _value_size(value) for name, value in item.items())


def check_item_size(
    namespace: str,
    doc_id: str,
    text: str | None,
    metadata: Metadata,
    gzip_threshold_bytes: int | None = None,
) -> None:
    """Raise :class:`ItemTooLargeError` if the document would exceed the item limit.

    Called before any write so an oversized document fails the whole upsert up
    front, instead of DynamoDB rejecting it partway through a batch.
    """
    _check_built_item(_build_item(namespace, doc_id, text, metadata, gzip_threshold_bytes))


def _check_built_item(item: dict[str, Any]) -> None:
    size = item_size_bytes(item)
    if size > MAX_ITEM_BYTES:
        raise ItemTooLargeError(item["id"], item["ns"], size, MAX_ITEM_BYTES)


def _read_text(item: dict[str, Any]) -> str | None:
    """Return an item's text, decompressing it if it was stored gzipped."""
    raw_text = item.get("text")
    gzip_blob = item.get("text_gzip")
    if gzip_blob is not None:
        return gzip.decompress(bytes(gzip_blob)).decode("utf-8")
    if isinstance(raw_text, (bytes, bytearray)):
        return gzip.decompress(bytes(raw_text)).decode("utf-8")
    return raw_text


class DynamoDBStore:
    """Thin, dependency-light wrapper over a single DynamoDB table."""

    _logger = logging.getLogger("dynavec.stores.dynamodb")

    def __init__(self, config: DynavecConfig, boto_session: Any | None = None) -> None:
        import boto3  # local import: base import stays cheap

        session = boto_session or boto3.Session()
        self._config = config
        resource_kwargs: dict[str, object] = {"region_name": config.region}
        botocore_config = config.botocore_config()
        if botocore_config is not None:
            resource_kwargs["config"] = botocore_config
        self._ddb = session.resource("dynamodb", **resource_kwargs)
        self._table = self._ddb.Table(config.table)
        self._logger = logging.getLogger("dynavec.stores.dynamodb")

    @staticmethod
    def _pk(namespace: str, doc_id: str) -> str:
        return _pk(namespace, doc_id)

    def put_many(
        self,
        namespace: str,
        items: list[tuple[str, str | None, Metadata]],
    ) -> None:
        """Upsert (id, text, metadata) triples. Uses batch writer (auto-retry).

        Raises :class:`ItemTooLargeError` before writing anything if any item is
        over DynamoDB's 400 KB limit.
        """
        t0 = time.perf_counter()
        threshold = self._config.gzip_threshold_bytes
        built = [
            _build_item(namespace, doc_id, text, metadata, threshold)
            for doc_id, text, metadata in items
        ]
        for item in built:
            _check_built_item(item)
        with self._table.batch_writer(overwrite_by_pkeys=["pk"]) as batch:
            for item in built:
                batch.put_item(Item=item)

        log_store_event(
            self._logger,
            "dynamodb.put_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            count=len(items),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def put_versioned(
        self,
        namespace: str,
        doc_id: str,
        text: str | None,
        metadata: Metadata,
        expected_version: int,
    ) -> int:
        """Write one document only if its stored version is ``expected_version``.

        The write stores ``expected_version + 1`` and returns it. Version 0 means
        "no version yet": the document is missing or was last written by
        :meth:`put_many`. Raises :class:`ConflictError` if another writer changed
        the document since it was read; nothing is written in that case.
        """
        from botocore.exceptions import ClientError

        t0 = time.perf_counter()
        item = _build_item(namespace, doc_id, text, metadata, self._config.gzip_threshold_bytes)
        _check_built_item(item)
        new_version = expected_version + 1
        item[VERSION_ATTR] = new_version

        condition: dict[str, Any] = {"ExpressionAttributeNames": {"#v": VERSION_ATTR}}
        if expected_version == 0:
            condition["ConditionExpression"] = "attribute_not_exists(#v)"
        else:
            condition["ConditionExpression"] = "#v = :expected"
            condition["ExpressionAttributeValues"] = {":expected": expected_version}

        try:
            self._table.put_item(Item=item, **condition)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise ConflictError(doc_id, namespace, expected_version) from exc
            raise

        log_store_event(
            self._logger,
            "dynamodb.put_versioned",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            version=new_version,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return new_version

    @retry()
    def get_versioned(self, namespace: str, doc_id: str) -> dict[str, Any] | None:
        """Strongly consistent read of one document, including its version.

        Returns ``{"text":..., "metadata":..., "version": int}`` or ``None``.
        """
        resp = self._table.get_item(Key={"pk": self._pk(namespace, doc_id)}, ConsistentRead=True)
        item = resp.get("Item")
        if item is None:
            return None
        return {
            "text": _read_text(item),
            "metadata": _from_dynamo(item.get("metadata", {})),
            "version": int(item.get(VERSION_ATTR, 0)),
        }

    @retry()
    def get_many(self, namespace: str, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate documents by id. Returns ``{id: {"text":..., "metadata":...}}``."""
        t0 = time.perf_counter()
        if not ids:
            log_store_event(
                self._logger,
                "dynamodb.get_many",
                self._config.structured_logging,
                table=self._config.table,
                namespace=namespace,
                requested_count=0,
                returned_count=0,
                duration_ms=0.0,
            )
            return {}
        keys = [{"pk": self._pk(namespace, doc_id)} for doc_id in ids]
        out: dict[str, dict[str, Any]] = {}

        for start in range(0, len(keys), _BATCH_GET_LIMIT):
            chunk = keys[start : start + _BATCH_GET_LIMIT]
            request: dict[str, Any] | None = {self._config.table: {"Keys": chunk}}
            while request:
                resp = self._ddb.batch_get_item(RequestItems=request)
                for item in resp["Responses"].get(self._config.table, []):
                    out[item["id"]] = {
                        "text": _read_text(item),
                        "metadata": _from_dynamo(item.get("metadata", {})),
                    }
                unprocessed = resp.get("UnprocessedKeys") or {}
                request = unprocessed if unprocessed else None

        log_store_event(
            self._logger,
            "dynamodb.get_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            requested_count=len(ids),
            returned_count=len(out),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return out

    def delete_many(self, namespace: str, ids: list[str]) -> None:
        t0 = time.perf_counter()
        with self._table.batch_writer() as batch:
            for doc_id in ids:
                batch.delete_item(Key={"pk": self._pk(namespace, doc_id)})

        log_store_event(
            self._logger,
            "dynamodb.delete_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            count=len(ids),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
