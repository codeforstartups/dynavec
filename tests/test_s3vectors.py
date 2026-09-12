from unittest.mock import MagicMock

import pytest

from dynavec.config import DynavecConfig
from dynavec.stores.s3vectors import _MAX_TOP_K, S3VectorsStore


class FakeS3VectorsClient:
	def __init__(self):
		self.calls = []

	def get_vectors(self, **kwargs):
		self.calls.append(kwargs)
		return {
			"vectors": [
				{"key": key, "data": {"float32": [1.0, 2.0]}}
				for key in kwargs["keys"]
			]
		}


def test_get_vectors_batches_keys_and_merges_results():
	fake = FakeS3VectorsClient()
	store = S3VectorsStore.__new__(S3VectorsStore)
	store._config = DynavecConfig(vector_bucket="bucket", index="index", table="table", dimension=2)
	store._client = fake

	keys = [f"key-{i}" for i in range(250)]
	result = store.get_vectors(keys)

	assert [len(call["keys"]) for call in fake.calls] == [100, 100, 50]
	assert all(call["returnData"] is True for call in fake.calls)
	assert len(result) == 250
	assert set(result) == set(keys)


def _config(**kwargs):
	defaults = dict(vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4)
	defaults.update(kwargs)
	return DynavecConfig(**defaults)


def _store_with_pages(pages, config=None):
	store = S3VectorsStore.__new__(S3VectorsStore)
	store._config = config or _config()
	store._client = MagicMock()
	paginator = MagicMock()
	paginator.paginate.return_value = pages
	store._client.get_paginator.return_value = paginator
	return store, paginator


def test_query_single_page():
	page = [{"key": f"k{i}", "distance": 0.01 * i} for i in range(50)]
	store, paginator = _store_with_pages([{"vectors": page}])

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

	assert len(results) == 50
	assert results[0]["key"] == "k0"
	assert results[-1]["key"] == "k49"
	store._client.get_paginator.assert_called_once_with("query_vectors")
	assert paginator.paginate.call_args.kwargs["PaginationConfig"] == {"MaxItems": 50}
	assert paginator.paginate.call_args.kwargs["topK"] == 50


def test_query_multi_page_pagination():
	pages = [
		{"vectors": [{"key": f"k{i}"} for i in range(100)]},
		{"vectors": [{"key": f"k{i}"} for i in range(100, 200)]},
		{"vectors": [{"key": f"k{i}"} for i in range(200, 250)]},
	]
	store, paginator = _store_with_pages(pages)

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=250)

	assert len(results) == 250
	assert results[0]["key"] == "k0"
	assert results[99]["key"] == "k99"
	assert results[100]["key"] == "k100"
	assert results[249]["key"] == "k249"
	assert paginator.paginate.call_args.kwargs["PaginationConfig"] == {"MaxItems": 250}


def test_query_truncation_at_top_k():
	pages = [
		{"vectors": [{"key": f"k{i}"} for i in range(100)]},
		{"vectors": [{"key": f"k{i}"} for i in range(100, 200)]},
	]
	store, _ = _store_with_pages(pages)

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=130)

	assert len(results) == 130
	assert results[-1]["key"] == "k129"


def test_query_fewer_results_than_top_k():
	store, _ = _store_with_pages([{"vectors": [{"key": "k0"}, {"key": "k1"}]}])

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

	assert len(results) == 2


def test_query_zero_or_negative_top_k():
	store, _ = _store_with_pages([{"vectors": [{"key": "k0"}]}])

	assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=0) == []
	assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=-5) == []
	store._client.get_paginator.assert_not_called()


def test_query_exceeding_service_limit():
	store, _ = _store_with_pages([])

	with pytest.raises(ValueError, match="exceeds Amazon S3 Vectors maximum limit"):
		store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=_MAX_TOP_K + 1)


def test_query_pages_raw_chunks():
	pages = [
		{"vectors": [{"key": f"k{i}"} for i in range(100)]},
		{"vectors": [{"key": f"k{i}"} for i in range(100, 150)]},
	]
	store, _ = _store_with_pages(pages)

	result_pages = list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=150))

	assert [len(p) for p in result_pages] == [100, 50]


def test_query_pages_configurable_page_size():
	store, _ = _store_with_pages(
		[{"vectors": [{"key": f"k{i}"} for i in range(25)]}]
	)

	result_pages = list(
		store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=25, page_size=10)
	)

	assert [len(p) for p in result_pages] == [10, 10, 5]


def test_query_pages_uses_config_top_k_page_size():
	store, _ = _store_with_pages(
		[{"vectors": [{"key": f"k{i}"} for i in range(25)]}],
		config=_config(top_k_page_size=10),
	)

	result_pages = list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=25))

	assert [len(p) for p in result_pages] == [10, 10, 5]


def test_query_pages_invalid_page_size():
	store, _ = _store_with_pages([])

	with pytest.raises(ValueError, match="page_size must be a positive integer"):
		list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=10, page_size=0))


def test_config_rejects_non_positive_top_k_page_size():
	with pytest.raises(ValueError, match="top_k_page_size must be a positive integer"):
		_config(top_k_page_size=0)
