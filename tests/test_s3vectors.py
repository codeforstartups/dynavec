"""Unit tests for S3VectorsStore pagination, service limits, and page sizing."""

from unittest.mock import MagicMock

import pytest

from dynavec.config import DynavecConfig
from dynavec.stores.s3vectors import _MAX_TOP_K, S3VectorsStore


@pytest.fixture
def store():
    cfg = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
    )
    mock_session = MagicMock()
    return S3VectorsStore(cfg, boto_session=mock_session)


def test_query_single_page(store):
    """When results fit in a single page (<100), returns them directly."""
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"vectors": [{"key": f"k{i}", "distance": 0.01 * i} for i in range(50)]}
    ]
    store._client.get_paginator.return_value = mock_paginator

    results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

    assert len(results) == 50
    assert results[0]["key"] == "k0"
    assert results[-1]["key"] == "k49"
    store._client.get_paginator.assert_called_once_with("query_vectors")


def test_query_multi_page_pagination(store):
    """When top_k > 100, drains multiple pages across nextTokens."""
    mock_paginator = MagicMock()
    page1 = [{"key": f"k{i}"} for i in range(100)]
    page2 = [{"key": f"k{i}"} for i in range(100, 200)]
    page3 = [{"key": f"k{i}"} for i in range(200, 250)]
    mock_paginator.paginate.return_value = [
        {"vectors": page1},
        {"vectors": page2},
        {"vectors": page3},
    ]
    store._client.get_paginator.return_value = mock_paginator

    results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=250)

    assert len(results) == 250
    assert results[0]["key"] == "k0"
    assert results[99]["key"] == "k99"
    assert results[100]["key"] == "k100"
    assert results[249]["key"] == "k249"


def test_query_truncation_at_top_k(store):
    """Ensures results are strictly truncated to top_k if a page yields excess items."""
    mock_paginator = MagicMock()
    page1 = [{"key": f"k{i}"} for i in range(100)]
    page2 = [{"key": f"k{i}"} for i in range(100, 200)]
    mock_paginator.paginate.return_value = [{"vectors": page1}, {"vectors": page2}]
    store._client.get_paginator.return_value = mock_paginator

    results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=130)

    assert len(results) == 130
    assert results[-1]["key"] == "k129"


def test_query_fewer_results_than_top_k(store):
    """When index has fewer vectors than requested top_k, returns all available without hanging."""
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"vectors": [{"key": "k0"}, {"key": "k1"}]}]
    store._client.get_paginator.return_value = mock_paginator

    results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

    assert len(results) == 2


def test_query_zero_or_negative_top_k(store):
    """Returns empty list immediately without issuing API calls."""
    assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=0) == []
    assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=-5) == []
    store._client.get_paginator.assert_not_called()


def test_query_exceeding_service_limit(store):
    """Raises ValueError if top_k exceeds S3 Vectors ceiling (10,000)."""
    with pytest.raises(ValueError, match="exceeds Amazon S3 Vectors maximum limit"):
        store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=_MAX_TOP_K + 1)


def test_query_pages_raw_chunks(store):
    """query_pages yields raw pages when page_size is None."""
    mock_paginator = MagicMock()
    page1 = [{"key": f"k{i}"} for i in range(100)]
    page2 = [{"key": f"k{i}"} for i in range(100, 150)]
    mock_paginator.paginate.return_value = [{"vectors": page1}, {"vectors": page2}]
    store._client.get_paginator.return_value = mock_paginator

    pages = list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=150))

    assert len(pages) == 2
    assert len(pages[0]) == 100
    assert len(pages[1]) == 50


def test_query_pages_configurable_page_size(store):
    """query_pages yields chunks corresponding to the requested page_size."""
    mock_paginator = MagicMock()
    page1 = [{"key": f"k{i}"} for i in range(25)]
    mock_paginator.paginate.return_value = [{"vectors": page1}]
    store._client.get_paginator.return_value = mock_paginator

    pages = list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=25, page_size=10))

    assert len(pages) == 3
    assert len(pages[0]) == 10
    assert len(pages[1]) == 10
    assert len(pages[2]) == 5


def test_query_pages_invalid_page_size(store):
    """Raises ValueError when page_size is <= 0."""
    with pytest.raises(ValueError, match="page_size must be a positive integer"):
        list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=10, page_size=0))
