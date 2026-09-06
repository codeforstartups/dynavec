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

from decimal import Decimal
from typing import Any

from ..config import DynavecConfig
from ..utils import retry

Metadata = dict[str, Any]

_BATCH_GET_LIMIT = 100


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


class DynamoDBStore:
    """Thin, dependency-light wrapper over a single DynamoDB table."""

    def __init__(self, config: DynavecConfig, boto_session=None) -> None:
        import boto3  # local import: base import stays cheap

        session = boto_session or boto3.Session()
        self._config = config
        self._ddb = session.resource("dynamodb", region_name=config.region)
        self._table = self._ddb.Table(config.table)

    @staticmethod
    def _pk(namespace: str, doc_id: str) -> str:
        return f"{namespace}#{doc_id}"

    def put_many(
        self,
        namespace: str,
        items: list[tuple[str, str | None, Metadata]],
    ) -> None:
        """Upsert (id, text, metadata) triples. Uses batch writer (auto-retry)."""
        with self._table.batch_writer(overwrite_by_pkeys=["pk"]) as batch:
            for doc_id, text, metadata in items:
                item = {
                    "pk": self._pk(namespace, doc_id),
                    "ns": namespace,
                    "id": doc_id,
                    "metadata": _to_dynamo(metadata or {}),
                }

                if text is not None:
                    item["text"] = text

                batch.put_item(Item=item)

    @retry()
    def get_many(
        self,
        namespace: str,
        ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Hydrate documents by id.

        Returns:
            A mapping of document id to text and metadata.
        """
        if not ids:
            return {}

        keys = [
            {"pk": self._pk(namespace, doc_id)}
            for doc_id in ids
        ]

        out: dict[str, dict[str, Any]] = {}

        for start in range(0, len(keys), _BATCH_GET_LIMIT):
            chunk = keys[start:start + _BATCH_GET_LIMIT]
            request = {
                self._config.table: {
                    "Keys": chunk
                }
            }

            while request:
                resp = self._ddb.batch_get_item(
                    RequestItems=request
                )

                for item in resp["Responses"].get(
                    self._config.table,
                    [],
                ):
                    out[item["id"]] = {
                        "text": item.get("text"),
                        "metadata": _from_dynamo(
                            item.get("metadata", {})
                        ),
                    }

                unprocessed = resp.get("UnprocessedKeys") or {}
                request = unprocessed if unprocessed else None

        return out

    def delete_many(
        self,
        namespace: str,
        ids: list[str],
    ) -> None:
        """Delete multiple documents from a namespace."""
        with self._table.batch_writer() as batch:
            for doc_id in ids:
                batch.delete_item(
                    Key={"pk": self._pk(namespace, doc_id)}
                )

    def list_namespaces(
        self,
        search: str | None = None,
    ) -> list[str]:
        """Return all namespaces containing documents.

        Namespace discovery is intended for observability and dashboard
        functionality, so a DynamoDB scan is acceptable here.
        """
        namespaces: set[str] = set()

        scan_kwargs: dict[str, Any] = {
            "ProjectionExpression": "#ns",
            "ExpressionAttributeNames": {
                "#ns": "ns",
            },
        }

        while True:
            response = self._table.scan(**scan_kwargs)

            for item in response.get("Items", []):
                namespace = item.get("ns")

                if namespace is not None:
                    namespaces.add(str(namespace))

            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

            scan_kwargs["ExclusiveStartKey"] = last_key

        results = sorted(namespaces)

        if search:
            needle = search.lower()

            results = [
                namespace
                for namespace in results
                if needle in namespace.lower()
            ]

        return results

    def namespace_stats(
        self,
        namespace: str,
    ) -> dict[str, int]:
        """Return document count and approximate text size for a namespace."""
        document_count = 0
        text_bytes = 0

        scan_kwargs: dict[str, Any] = {
            "ProjectionExpression": "#ns, #text",
            "FilterExpression": "#ns = :namespace",
            "ExpressionAttributeNames": {
                "#ns": "ns",
                "#text": "text",
            },
            "ExpressionAttributeValues": {
                ":namespace": namespace,
            },
        }

        while True:
            response = self._table.scan(**scan_kwargs)

            for item in response.get("Items", []):
                document_count += 1

                text = item.get("text")

                if text is not None:
                    text_bytes += len(
                        str(text).encode("utf-8")
                    )

            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

            scan_kwargs["ExclusiveStartKey"] = last_key

        return {
            "document_count": document_count,
            "text_bytes": text_bytes,
        }