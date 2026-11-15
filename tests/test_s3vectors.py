import time
from contextlib import closing
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.stub import Stubber

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
    store._put_limiter = None
    store._query_limiter = None

    store._client.query_vectors.side_effect = pages or [{"vectors": []}]
    return store, store._client.query_vectors


def test_query_single_page():
	page = [{"key": f"k{i}", "distance": 0.01 * i} for i in range(50)]
	store, query = _store_with_pages([{"vectors": page}])

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

	assert len(results) == 50
	assert results[0]["key"] == "k0"
	assert results[-1]["key"] == "k49"
	store._client.get_paginator.assert_not_called()
	assert query.call_args.kwargs["topK"] == 50
	assert "nextToken" not in query.call_args.kwargs


def test_query_with_real_sdk_client():
    session = boto3.Session(
        aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-1"
    )
    store = S3VectorsStore(_config(), session)
    vectors = [{"key": "doc-1", "distance": 0.25}]

    with closing(store._client), Stubber(store._client) as stubber:
        stubber.add_response("query_vectors", {"vectors": vectors}, {
            "vectorBucketName": "test-bucket", "indexName": "test-index", "topK": 1,
            "queryVector": {"float32": [1.0, 2.0, 3.0, 4.0]},
            "returnMetadata": True, "returnDistance": True,
        })

        assert store.query([1.0, 2.0, 3.0, 4.0], top_k=1) == vectors
        stubber.assert_no_pending_responses()

def test_query_multi_page_pagination():
    pages = [
        {
            "vectors": [{"key": f"k{i}"} for i in range(100)],
            "nextToken": "token-1",
        },
        {
            "vectors": [{"key": f"k{i}"} for i in range(100, 200)],
            "nextToken": "token-2",
        },
        {
            "vectors": [{"key": f"k{i}"} for i in range(200, 250)],
        },
    ]

    store, _ = _store_with_pages(pages)

    results = store.query(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=250,
    )

    assert len(results) == 250
    assert [result["key"] for result in results[:3]] == ["k0", "k1", "k2"]
    assert [result["key"] for result in results[-3:]] == [
        "k247",
        "k248",
        "k249",
    ]
    calls = store._client.query_vectors.call_args_list
    assert "nextToken" not in calls[0].kwargs
    assert [call.kwargs["nextToken"] for call in calls[1:]] == ["token-1", "token-2"]

def test_query_fewer_results_than_top_k():
	store, _ = _store_with_pages([{"vectors": [{"key": "k0"}, {"key": "k1"}]}])

	results = store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=50)

	assert len(results) == 2


def test_query_zero_or_negative_top_k():
	store, _ = _store_with_pages([{"vectors": [{"key": "k0"}]}])

	assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=0) == []
	assert store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=-5) == []
	store._client.query_vectors.assert_not_called()


def test_query_exceeding_service_limit():
	store, _ = _store_with_pages([])

	with pytest.raises(ValueError, match="exceeds Amazon S3 Vectors maximum limit"):
		store.query(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=_MAX_TOP_K + 1)


def test_query_pages_invalid_page_size():
	store, _ = _store_with_pages([])

	with pytest.raises(ValueError, match="page_size must be a positive integer"):
		list(store.query_pages(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=10, page_size=0))


def test_config_rejects_non_positive_top_k_page_size():
	with pytest.raises(ValueError, match="top_k_page_size must be a positive integer"):
		_config(top_k_page_size=0)

def test_put_vectors_parallelization():
    store, _ = _store_with_pages([])

    vectors = [
        (f"key_{i}", [0.1] * 128, {"tag": "test"})
        for i in range(1500)
    ]

    def mock_put_batch(payload):
        time.sleep(0.1)

    store._put_batch = MagicMock(side_effect=mock_put_batch)

    t0 = time.perf_counter()
    store.put_vectors(vectors)
    duration = time.perf_counter() - t0

    assert duration < 0.5
    assert store._put_batch.call_count == 3


def test_put_vectors_error_propagation():
    store, _ = _store_with_pages([])

    vectors = [
        (f"key_{i}", [0.1] * 128, {"tag": "test"})
        for i in range(1000)
    ]

    def mock_put_batch_with_error(payload):
        if payload[0]["key"] == "key_0":
            raise RuntimeError("S3 API Failure")

    store._put_batch = MagicMock(side_effect=mock_put_batch_with_error)

    with pytest.raises(RuntimeError, match="S3 API Failure"):
        store.put_vectors(vectors)


def test_put_batch_uses_rate_limiter():
    store, _ = _store_with_pages(
        [],
        config=_config(put_rps=10),
    )

    limiter = MagicMock()
    store._put_limiter = limiter

    payload = [
        {
            "key": "key-1",
            "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
            "metadata": {},
        }
    ]

    store._client.put_vectors = MagicMock()

    store._put_batch(payload)

    limiter.acquire.assert_called_once_with()
    store._client.put_vectors.assert_called_once()


def test_query_truncation_at_top_k():
    pages = [
        {
            "vectors": [{"key": f"k{i}"} for i in range(100)],
            "nextToken": "token-1",
        },
        {
            "vectors": [{"key": f"k{i}"} for i in range(100, 200)],
        },
    ]

    store, _ = _store_with_pages(pages)

    results = store.query(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=130,
    )

    assert len(results) == 130
    assert results[-1]["key"] == "k129"


def test_query_pages_raw_chunks():
    pages = [
        {
            "vectors": [{"key": f"k{i}"} for i in range(100)],
            "nextToken": "token-1",
        },
        {
            "vectors": [{"key": f"k{i}"} for i in range(100, 150)],
        },
    ]

    store, _ = _store_with_pages(pages)

    result_pages = list(
        store.query_pages(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=150,
        )
    )

    assert [len(p) for p in result_pages] == [100, 50]


def test_query_pages_configurable_page_size():
    store, _ = _store_with_pages(
        [{"vectors": [{"key": f"k{i}"} for i in range(25)]}]
    )

    result_pages = list(
        store.query_pages(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=25,
            page_size=10,
        )
    )

    assert [len(p) for p in result_pages] == [10, 10, 5]


def test_query_pages_uses_config_top_k_page_size():
    store, _ = _store_with_pages(
        [{"vectors": [{"key": f"k{i}"} for i in range(25)]}],
        config=_config(top_k_page_size=10),
    )

    result_pages = list(
        store.query_pages(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=25,
        )
    )

    assert [len(p) for p in result_pages] == [10, 10, 5]


@pytest.mark.parametrize("page_size", [None, 2])
def test_query_pages_continues_after_empty_page_with_next_token(page_size):
    store, _ = _store_with_pages([
        {"vectors": [{"key": "k0"}], "nextToken": "token-1"},
        {"vectors": [], "nextToken": "token-2"},
        {"vectors": [{"key": "k1"}, {"key": "k2"}]},
    ])
    limiter = MagicMock()
    store._query_limiter = limiter

    pages = list(store.query_pages([0.1, 0.2, 0.3, 0.4], top_k=5, page_size=page_size))

    assert [hit["key"] for page in pages for hit in page] == ["k0", "k1", "k2"]
    assert [len(page) for page in pages] == ([1, 2] if page_size is None else [2, 1])
    assert limiter.acquire.call_count == 3


def test_query_pages_stops_at_top_k_with_large_page_size():
    store, query = _store_with_pages([
        {"vectors": [{"key": "k0"}], "nextToken": "token-1"},
        {"vectors": [{"key": "k1"}, {"key": "k2"}], "nextToken": "unused"},
    ])

    pages = list(store.query_pages(
        [1.0, 2.0, 3.0, 4.0], top_k=2, page_size=100,
        filter={"kind": "test"}, return_metadata=False, return_distance=False,
    ))

    assert pages == [[{"key": "k0"}, {"key": "k1"}]]
    assert query.call_count == 2
    for call in query.call_args_list:
        assert call.kwargs["topK"] == 2
        assert call.kwargs["filter"] == {"kind": "test"}
        assert call.kwargs["returnMetadata"] is False
        assert call.kwargs["returnDistance"] is False


def test_query_pages_uses_rate_limiter_per_page():
    pages = [
        {
            "vectors": [{"key": "k0"}],
            "nextToken": "token-1",
        },
        {
            "vectors": [{"key": "k1"}],
            "nextToken": "token-2",
        },
        {
            "vectors": [{"key": "k2"}],
        },
    ]

    store, _ = _store_with_pages(
        pages,
        config=_config(query_rps=10),
    )

    limiter = MagicMock()
    store._query_limiter = limiter

    result_pages = list(
        store.query_pages(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=3,
        )
    )

    assert [len(page) for page in result_pages] == [1, 1, 1]
    assert limiter.acquire.call_count == 3


def test_query_pages_does_not_rate_limit_after_paginator_exhaustion():
    store, _ = _store_with_pages(
        [{"vectors": [{"key": "k0"}]}],
        config=_config(query_rps=10),
    )

    limiter = MagicMock()
    store._query_limiter = limiter

    result_pages = list(
        store.query_pages(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=50,
        )
    )

    assert [len(page) for page in result_pages] == [1]
    assert limiter.acquire.call_count == 1
