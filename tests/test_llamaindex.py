import sys
from unittest.mock import MagicMock

import pytest

if sys.version_info < (3, 10):
    pytest.skip(
        "LlamaIndex test dependencies are not compatible with Python 3.9",
        allow_module_level=True,
    )

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from dynavec.integrations.llamaindex import (
    DynavecLlamaStore,
    _translate_filter,
    _translate_filters,
)


@pytest.mark.parametrize(
    ("operator", "value", "expected_operator"),
    [
        (FilterOperator.EQ, "ai", "$eq"),
        (FilterOperator.NE, "ai", "$ne"),
        (FilterOperator.GT, 10, "$gt"),
        (FilterOperator.GTE, 10, "$gte"),
        (FilterOperator.LT, 10, "$lt"),
        (FilterOperator.LTE, 10, "$lte"),
        (FilterOperator.IN, ["ai", "ml"], "$in"),
        (FilterOperator.NIN, ["ai", "ml"], "$nin"),
    ],
)
def test_translate_filter(operator, value, expected_operator):
    metadata_filter = MetadataFilter(
        key="field",
        value=value,
        operator=operator,
    )

    assert _translate_filter(metadata_filter) == {
        "field": {
            expected_operator: value,
        }
    }


def test_translate_filters_and():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value="ai",
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="year",
                value=2024,
                operator=FilterOperator.GTE,
            ),
        ],
        condition=FilterCondition.AND,
    )

    assert _translate_filters(filters) == {
        "$and": [
            {"category": {"$eq": "ai"}},
            {"year": {"$gte": 2024}},
        ]
    }


def test_translate_filters_or():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="ai"),
            MetadataFilter(key="category", value="ml"),
        ],
        condition=FilterCondition.OR,
    )

    assert _translate_filters(filters) == {
        "$or": [
            {"category": {"$eq": "ai"}},
            {"category": {"$eq": "ml"}},
        ]
    }


def test_translate_nested_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value="ai",
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="year",
                        value=2024,
                        operator=FilterOperator.GTE,
                    ),
                    MetadataFilter(
                        key="status",
                        value="featured",
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )

    assert _translate_filters(filters) == {
        "$and": [
            {"category": {"$eq": "ai"}},
            {
                "$or": [
                    {"year": {"$gte": 2024}},
                    {"status": {"$eq": "featured"}},
                ]
            },
        ]
    }


def test_translate_empty_filters():
    filters = MetadataFilters(filters=[])

    assert _translate_filters(filters) == {}


def test_unsupported_filter_operator():
    metadata_filter = MetadataFilter(
        key="title",
        value="python",
        operator=FilterOperator.TEXT_MATCH,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported LlamaIndex metadata filter operator",
    ):
        _translate_filter(metadata_filter)


@pytest.mark.skipif(
    not hasattr(FilterCondition, "NOT"),
    reason="FilterCondition.NOT is unavailable in older LlamaIndex versions",
)
def test_unsupported_not_condition():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="ai"),
        ],
        condition=FilterCondition.NOT,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported LlamaIndex metadata filter condition",
    ):
        _translate_filters(filters)


def test_query_passes_translated_filters_to_dynavec():
    client = MagicMock()
    client.search.return_value = []

    store = DynavecLlamaStore(client, namespace="kb")

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value="ai",
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="year",
                value=2024,
                operator=FilterOperator.GTE,
            ),
        ],
        condition=FilterCondition.AND,
    )

    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=3,
        filters=filters,
    )

    result = store.query(query)

    client.search.assert_called_once_with(
        vector=[0.1, 0.2],
        top_k=3,
        namespace="kb",
        filter={
            "$and": [
                {"category": {"$eq": "ai"}},
                {"year": {"$gte": 2024}},
            ]
        },
    )

    assert result.nodes == []
    assert result.ids == []
    assert result.similarities == []
