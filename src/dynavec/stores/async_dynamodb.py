"""Async DynamoDB document / metadata store backed by aioboto3."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from ..config import DynavecConfig
from ..logging import log_store_event
from ..utils import async_retry
from .dynamodb import (
    _BATCH_GET_LIMIT,
    Metadata,
    _build_item,
    _check_built_item,
    _from_dynamo,
    _pk,
    _read_text,
)


class AsyncDynamoDBStore:
    """Async wrapper around DynamoDB document and metadata I/O."""

    _logger = logging.getLogger("dynavec.stores.async_dynamodb")

    def __init__(self, config: DynavecConfig, boto_session: Any) -> None:
        self._config = config
        self._session = boto_session
        self._resource_context: Any | None = None
        self._ddb: Any | None = None
        self._table: Any | None = None

    async def __aenter__(self) -> AsyncDynamoDBStore:
        if self._ddb is not None:
            return self

        resource_kwargs: dict[str, object] = {
            "region_name": self._config.region,
        }

        botocore_config = self._config.botocore_config()
        if botocore_config is not None:
            resource_kwargs["config"] = botocore_config

        self._resource_context = self._session.resource(
            "dynamodb",
            **resource_kwargs,
        )

        try:
            self._ddb = await self._resource_context.__aenter__()
            self._table = await self._ddb.Table(self._config.table)
        except Exception:
            try:
                await self._resource_context.__aexit__(
                    None,
                    None,
                    None,
                )
            finally:
                self._table = None
                self._ddb = None
                self._resource_context = None
            raise

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._resource_context is not None:
                await self._resource_context.__aexit__(
                    exc_type,
                    exc_value,
                    traceback,
                )
        finally:
            self._table = None
            self._ddb = None
            self._resource_context = None

    async def aclose(self) -> None:
        try:
            if self._resource_context is not None:
                await self._resource_context.__aexit__(
                    None,
                    None,
                    None,
                )
        finally:
            self._table = None
            self._ddb = None
            self._resource_context = None

    @staticmethod
    def _pk(namespace: str, doc_id: str) -> str:
        return _pk(namespace, doc_id)

    def _require_ddb(self) -> Any:
        if self._ddb is None:
            raise RuntimeError(
                "AsyncDynamoDBStore is not open. "
                "Use it inside 'async with' before performing AWS operations."
            )
        return self._ddb

    def _require_table(self) -> Any:
        if self._table is None:
            raise RuntimeError(
                "AsyncDynamoDBStore is not open. "
                "Use it inside 'async with' before performing AWS operations."
            )
        return self._table

    async def put_many(
        self,
        namespace: str,
        items: Sequence[
            tuple[str, str | None, Metadata] | tuple[str, str | None, Metadata, int | None]
        ],
    ) -> None:
        """Upsert document tuples using DynamoDB's async batch writer."""
        t0 = time.perf_counter()

        threshold = self._config.gzip_threshold_bytes
        ttl_attr = self._config.dynamodb_ttl_attribute

        built = [
            _build_item(
                namespace,
                item[0],
                item[1],
                item[2],
                threshold,
                ttl=item[3] if len(item) > 3 else None,
                ttl_attribute=ttl_attr,
            )
            for item in items
        ]

        for item in built:
            _check_built_item(item)

        table = self._require_table()

        async with table.batch_writer(
            overwrite_by_pkeys=["pk"],
        ) as batch:
            for item in built:
                await batch.put_item(Item=item)

        log_store_event(
            self._logger,
            "dynamodb.put_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            count=len(items),
            duration_ms=round(
                (time.perf_counter() - t0) * 1000,
                2,
            ),
        )

    @async_retry()
    async def get_many(
        self,
        namespace: str,
        ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Hydrate documents by id asynchronously."""
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

        ddb = self._require_ddb()

        keys = [{"pk": self._pk(namespace, doc_id)} for doc_id in ids]

        out: dict[str, dict[str, Any]] = {}

        for start in range(0, len(keys), _BATCH_GET_LIMIT):
            chunk = keys[start : start + _BATCH_GET_LIMIT]

            request: dict[str, Any] | None = {
                self._config.table: {
                    "Keys": chunk,
                }
            }

            while request:
                response = await ddb.batch_get_item(
                    RequestItems=request,
                )

                for item in response["Responses"].get(
                    self._config.table,
                    [],
                ):
                    entry: dict[str, Any] = {
                        "text": _read_text(item),
                        "metadata": _from_dynamo(item.get("metadata", {})),
                    }

                    ttl_attr = self._config.dynamodb_ttl_attribute
                    if ttl_attr in item:
                        entry["ttl"] = int(item[ttl_attr])

                    out[item["id"]] = entry

                unprocessed = response.get("UnprocessedKeys") or {}

                request = unprocessed if unprocessed else None

        log_store_event(
            self._logger,
            "dynamodb.get_many",
            self._config.structured_logging,
            table=self._config.table,
            namespace=namespace,
            requested_count=len(ids),
            returned_count=len(out),
            duration_ms=round(
                (time.perf_counter() - t0) * 1000,
                2,
            ),
        )

        return out
