"""Tests for the AsyncDynavec client."""

import pytest
import sys
from unittest.mock import AsyncMock, patch, MagicMock

# Mock aioboto3 globally to bypass PyPI installation timeout
sys.modules["aioboto3"] = MagicMock()

from dynavec.async_client import AsyncDynavec
from dynavec.config import DynavecConfig
from dynavec.models import Document, SearchResult, UpsertResult

@pytest.fixture
def config():
    return DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=1536,
    )

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.dimension = 1536
    embedder.embed_documents.return_value = [[0.1] * 1536]
    return embedder

@pytest.fixture
def mock_aioboto_session():
    return MagicMock()

@pytest.mark.asyncio
async def test_async_search(config, mock_embedder, mock_aioboto_session):
    with patch("dynavec.async_client.AsyncS3VectorsStore") as mock_s3, \
         patch("dynavec.async_client.AsyncDynamoDBStore") as mock_ddb:
        
        # Setup mocks
        mock_s3_instance = mock_s3.return_value
        mock_ddb_instance = mock_ddb.return_value
        
        mock_s3_instance.query = AsyncMock(return_value=[
            {"key": "default#doc1", "distance": 0.05}
        ])
        
        mock_ddb_instance.get_many = AsyncMock(return_value={
            "doc1": {"text": "hello world", "metadata": {"lang": "en"}}
        })
        
        client = AsyncDynavec(config, embedder=mock_embedder, aioboto_session=mock_aioboto_session)
        results = await client.search("hello", top_k=1)
        
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].text == "hello world"
        assert results[0].metadata == {"lang": "en"}
        
        # Verify calls
        mock_s3_instance.query.assert_called_once()
        mock_ddb_instance.get_many.assert_called_once_with("default", ["doc1"])
        mock_embedder.embed_documents.assert_called_once_with(["hello"])

@pytest.mark.asyncio
async def test_async_upsert(config, mock_embedder, mock_aioboto_session):
    with patch("dynavec.async_client.AsyncS3VectorsStore") as mock_s3, \
         patch("dynavec.async_client.AsyncDynamoDBStore") as mock_ddb:
        
        mock_s3_instance = mock_s3.return_value
        mock_ddb_instance = mock_ddb.return_value
        
        mock_s3_instance.put_vectors = AsyncMock()
        mock_ddb_instance.put_many = AsyncMock()
        
        client = AsyncDynavec(config, embedder=mock_embedder, aioboto_session=mock_aioboto_session)
        
        docs = [Document(id="doc2", text="async rocks", metadata={"async": True})]
        result = await client.upsert(docs)
        
        assert result.count == 1
        assert result.ids == ["doc2"]
        
        mock_embedder.embed_documents.assert_called_once_with(["async rocks"])
        mock_s3_instance.put_vectors.assert_called_once()
        mock_ddb_instance.put_many.assert_called_once()
