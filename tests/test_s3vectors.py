from dynavec.config import DynavecConfig
from dynavec.stores.s3vectors import S3VectorsStore


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