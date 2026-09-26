"""Semantic Kernel vector store adapter for dynavec."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any, Generic

from ..client import Dynavec
from ..exceptions import MissingDependencyError
from ..models import Document as DVDocument

try:
    from semantic_kernel.data.vector import (
        KernelSearchResults,
        SearchType,
        TKey,
        TModel,
        VectorSearch,
        VectorSearchOptions,
        VectorSearchResult,
        VectorStore,
        VectorStoreCollection,
        VectorStoreCollectionDefinition,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise MissingDependencyError(
        "DynavecSemanticKernelStore",
        "semantic-kernel",
        "semantic-kernel",
    ) from exc


class DynavecCollection(
    VectorStoreCollection[TKey, TModel],
    VectorSearch[TKey, TModel],
    Generic[TKey, TModel],
):
    """Semantic Kernel collection backed by Dynavec."""

    supported_search_types = {SearchType.VECTOR}

    def __init__(
        self,
        client: Dynavec,
        record_type: type[TModel],
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            record_type=record_type,
            definition=definition,
            collection_name=collection_name,
            **kwargs,
        )
        self._client = client
        self._namespace = collection_name or "default"

    async def _inner_upsert(
        self,
        records: Sequence[Any],
        **kwargs: Any,
    ) -> Sequence[TKey]:
        result = self._client.upsert(
            list(records),
            namespace=self._namespace,
        )
        return result.ids

    async def _inner_get(
        self,
        keys: Sequence[TKey] | None = None,
        options=None,
        **kwargs: Any,
    ):
        if keys is None:
            raise NotImplementedError("Dynavec connector currently supports get by key only.")

        results = self._client.get(
            [str(key) for key in keys],
            namespace=self._namespace,
        )

        return [
            DVDocument(
                id=result.id,
                text=result.text,
                vector=result.vector,
                metadata=result.metadata,
            )
            for result in results
        ]

    async def _inner_delete(
        self,
        keys: Sequence[TKey],
        **kwargs: Any,
    ) -> None:
        self._client.delete(
            [str(key) for key in keys],
            namespace=self._namespace,
        )

    async def ensure_collection_exists(self, **kwargs: Any) -> None:
        return None

    async def collection_exists(self, **kwargs: Any) -> bool:
        return True

    async def ensure_collection_deleted(self, **kwargs: Any) -> None:
        return None

    async def _inner_search(
        self,
        search_type: SearchType,
        options: VectorSearchOptions,
        values: Any | None = None,
        vector: Sequence[float | int] | None = None,
        **kwargs: Any,
    ) -> KernelSearchResults[VectorSearchResult[TModel]]:
        search_filter = self._build_filter(options.filter)

        if vector is not None:
            results = self._client.search(
                vector=list(vector),
                top_k=options.skip + options.top,
                namespace=self._namespace,
                filter=search_filter,
                include_vectors=options.include_vectors,
            )
        else:
            results = self._client.search(
                query=values,
                top_k=options.skip + options.top,
                namespace=self._namespace,
                filter=search_filter,
                include_vectors=options.include_vectors,
            )

        results = results[options.skip : options.skip + options.top]

        return KernelSearchResults(
            results=self._get_vector_search_results_from_results(
                results,
                options,
            ),
        )

    def _get_record_from_result(self, result: Any) -> DVDocument:
        return DVDocument(
            id=result.id,
            text=result.text,
            vector=result.vector,
            metadata=result.metadata,
        )

    def _get_score_from_result(self, result: Any) -> float | None:
        return result.score

    def _serialize_dicts_to_store_models(
        self, records: list[dict[str, Any]], **kwargs: Any
    ) -> list[DVDocument]:
        key_name = self._key_field_storage_name

        vector_field = self.definition.vector_fields[0] if self.definition.vector_fields else None
        vector_name = vector_field.storage_name or vector_field.name if vector_field else None

        documents = []

        for record in records:
            record_id = str(record[key_name])
            vector = record.get(vector_name) if vector_name else None

            text = None
            metadata = {}

            for field in self.definition.data_fields:
                field_name = field.storage_name or field.name
                value = record.get(field_name)

                if field.name in {"text", "content"}:
                    text = value
                elif value is not None:
                    metadata[field_name] = value

            documents.append(
                DVDocument(
                    id=record_id,
                    text=text,
                    vector=vector,
                    metadata=metadata,
                )
            )

        return documents

    def _deserialize_store_models_to_dicts(
        self, records: list[DVDocument], **kwargs: Any
    ) -> list[dict[str, Any]]:
        result = []

        for record in records:
            data = {
                self._key_field_storage_name: record.id,
            }

            for field in self.definition.vector_fields:
                field_name = field.storage_name or field.name
                data[field_name] = record.vector

            for field in self.definition.data_fields:
                field_name = field.storage_name or field.name

                if field.name in {"text", "content"}:
                    data[field_name] = record.text
                else:
                    data[field_name] = record.metadata.get(field_name)

            result.append(data)

        return result

    def _lambda_parser(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [self._lambda_parser(element) for element in node.elts]

        if isinstance(node, ast.Subscript):
            return self._lambda_parser(node.slice)

        if isinstance(node, ast.Attribute):
            return node.attr

        if isinstance(node, ast.Compare):
            field = self._lambda_parser(node.left)
            filters = []

            for operator, comparator in zip(node.ops, node.comparators):
                value = self._lambda_parser(comparator)

                if isinstance(operator, ast.Eq):
                    filters.append({field: value})
                elif isinstance(operator, ast.NotEq):
                    filters.append({field: {"$ne": value}})
                elif isinstance(operator, ast.Gt):
                    filters.append({field: {"$gt": value}})
                elif isinstance(operator, ast.GtE):
                    filters.append({field: {"$gte": value}})
                elif isinstance(operator, ast.Lt):
                    filters.append({field: {"$lt": value}})
                elif isinstance(operator, ast.LtE):
                    filters.append({field: {"$lte": value}})
                elif isinstance(operator, ast.In):
                    filters.append({field: {"$in": value}})
                elif isinstance(operator, ast.NotIn):
                    filters.append({field: {"$nin": value}})
                else:
                    raise ValueError(f"Unsupported comparison operator: {type(operator).__name__}")

            return filters[0] if len(filters) == 1 else {"$and": filters}

        if isinstance(node, ast.BoolOp):
            values = [self._lambda_parser(value) for value in node.values]

            if isinstance(node.op, ast.And):
                return {"$and": values}

            if isinstance(node.op, ast.Or):
                return {"$or": values}

        raise ValueError(f"Unsupported filter expression: {type(node).__name__}")


class DynavecStore(VectorStore):
    """Semantic Kernel VectorStore backed by Dynavec."""

    def __init__(
        self,
        client: Dynavec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client

    def get_collection(
        self,
        record_type: type[TModel],
        *,
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> DynavecCollection:
        return DynavecCollection(
            client=self._client,
            record_type=record_type,
            definition=definition,
            collection_name=collection_name,
            **kwargs,
        )

    async def list_collection_names(self, **kwargs: Any) -> Sequence[str]:
        return []
