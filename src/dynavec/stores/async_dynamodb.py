"""Async DynamoDB document / metadata store."""

from __future__ import annotations

import gzip
import logging
import time
from typing import Any

from ..config import DynavecConfig
from ..logging import log_store_event
from .dynamodb import _build_item, _check_built_item, _from_dynamo, _pk

Metadata = dict[str, Any]
_BATCH_GET_LIMIT = 100

class AsyncDynamoDBStore:
    _logger = logging.getLogger("dynavec.stores.async_dynamodb")

    def __init__(self, config: DynavecConfig, aioboto_session=None) -> None:
        self._config = config
        if aioboto_session is None:
            import aioboto3
            aioboto_session = aioboto3.Session()
        self._session = aioboto_session
        self._resource_kwargs: dict[str, object] = {"region_name": config.region}
        botocore_config = config.botocore_config()
        if botocore_config is not None:
            self._resource_kwargs["config"] = botocore_config

    @staticmethod
    def _pk(namespace: str, doc_id: str) -> str:
        return _pk(namespace, doc_id)

    async def put_many(
        self,
        namespace: str,
        items: list[tuple[str, str | None, Metadata]],
    ) -> None:
        t0 = time.perf_counter()
        threshold = self._config.gzip_threshold_bytes
        built = [
            _build_item(namespace, doc_id, text, metadata, threshold)
            for doc_id, text, metadata in items
        ]
        for item in built:
            _check_built_item(item)
            
        async with self._session.resource("dynamodb", **self._resource_kwargs) as ddb:
            table = await ddb.Table(self._config.table)
            # aioboto3 doesn't have batch_writer out of the box like boto3, 
            # so we use a simple loop over put_item or batch_write_item.
            # Using simple put_item loop here for simplicity; production should batch.
            for item in built:
                await table.put_item(Item=item)

        log_store_event(
            self._logger,
            "async_dynamodb.put_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            count=len(items),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    async def get_many(self, namespace: str, ids: list[str]) -> dict[str, dict[str, Any]]:
        t0 = time.perf_counter()
        if not ids:
            return {}
            
        keys = [{"pk": self._pk(namespace, doc_id)} for doc_id in ids]
        out: dict[str, dict[str, Any]] = {}

        async with self._session.resource("dynamodb", **self._resource_kwargs) as ddb:
            for start in range(0, len(keys), _BATCH_GET_LIMIT):
                chunk = keys[start : start + _BATCH_GET_LIMIT]
                request = {self._config.table: {"Keys": chunk}}
                
                while request:
                    resp = await ddb.batch_get_item(RequestItems=request)
                    for item in resp["Responses"].get(self._config.table, []):
                        raw_text = item.get("text")
                        gzip_blob = item.get("text_gzip")
                        if gzip_blob is not None:
                            decompressed_text = gzip.decompress(bytes(gzip_blob)).decode("utf-8")
                        elif isinstance(raw_text, (bytes, bytearray)):
                            decompressed_text = gzip.decompress(bytes(raw_text)).decode("utf-8")
                        else:
                            decompressed_text = raw_text
                        out[item["id"]] = {
                            "text": decompressed_text,
                            "metadata": _from_dynamo(item.get("metadata", {})),
                        }
                    unprocessed = resp.get("UnprocessedKeys") or {}
                    request = unprocessed if unprocessed else None

        log_store_event(
            self._logger,
            "async_dynamodb.get_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            requested_count=len(ids),
            returned_count=len(out),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return out

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        t0 = time.perf_counter()
        async with self._session.resource("dynamodb", **self._resource_kwargs) as ddb:
            table = await ddb.Table(self._config.table)
            for doc_id in ids:
                await table.delete_item(Key={"pk": self._pk(namespace, doc_id)})

        log_store_event(
            self._logger,
            "async_dynamodb.delete_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            count=len(ids),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
