import gzip
from unittest.mock import MagicMock

import pytest

from dynavec.config import DynavecConfig
from dynavec.exceptions import ItemTooLargeError
from dynavec.stores.dynamodb import (
    MAX_ITEM_BYTES,
    DynamoDBStore,
    _build_item,
    check_item_size,
)


def test_config_validation():
    with pytest.raises(ValueError, match="gzip_threshold_bytes must be a positive integer"):
        DynavecConfig(
            vector_bucket="b",
            index="i",
            table="t",
            dimension=4,
            gzip_threshold_bytes=0,
        )

    with pytest.raises(ValueError, match="gzip_threshold_bytes must be a positive integer"):
        DynavecConfig(
            vector_bucket="b",
            index="i",
            table="t",
            dimension=4,
            gzip_threshold_bytes=-100,
        )


def test_sub_threshold_text_stays_plain_string():
    item = _build_item("ns", "doc1", "short text", {}, gzip_threshold_bytes=1000)
    assert "text" in item
    assert item["text"] == "short text"
    assert "text_gzip" not in item


def test_above_threshold_text_is_compressed():
    text = "repeated text " * 50
    item = _build_item("ns", "doc1", text, {}, gzip_threshold_bytes=50)
    assert "text_gzip" in item
    assert "text" not in item
    assert isinstance(item["text_gzip"], bytes)
    assert gzip.decompress(item["text_gzip"]).decode("utf-8") == text


def test_get_many_transparent_decompression():
    config = DynavecConfig(
        vector_bucket="b",
        index="i",
        table="docs",
        dimension=4,
        gzip_threshold_bytes=100,
    )
    session = MagicMock()
    ddb = session.resource.return_value

    original_compressed_text = "compressed document text " * 30
    compressed_bytes = gzip.compress(original_compressed_text.encode("utf-8"))

    ddb.batch_get_item.return_value = {
        "Responses": {
            "docs": [
                {
                    "id": "doc1",
                    "text_gzip": compressed_bytes,
                    "metadata": {"topic": "ai"},
                },
                {
                    "id": "doc2",
                    "text": "regular uncompressed text",
                    "metadata": {"topic": "search"},
                },
                {
                    "id": "doc3",
                    "text": compressed_bytes,  # binary text field fallback
                    "metadata": {"topic": "legacy"},
                },
            ]
        },
        "UnprocessedKeys": {},
    }

    store = DynamoDBStore(config, boto_session=session)
    hydrated = store.get_many("ns", ["doc1", "doc2", "doc3"])

    assert hydrated["doc1"]["text"] == original_compressed_text
    assert hydrated["doc1"]["metadata"] == {"topic": "ai"}

    assert hydrated["doc2"]["text"] == "regular uncompressed text"
    assert hydrated["doc2"]["metadata"] == {"topic": "search"}

    assert hydrated["doc3"]["text"] == original_compressed_text
    assert hydrated["doc3"]["metadata"] == {"topic": "legacy"}


def test_large_compressible_text_bypasses_400kb_limit():
    # 500 KB of repetitive text that compresses down to ~2 KB
    large_text = "Dynavec is a serverless vector database on AWS. " * 11000
    assert len(large_text.encode("utf-8")) > MAX_ITEM_BYTES

    # Without compression -> raises ItemTooLargeError
    with pytest.raises(ItemTooLargeError):
        check_item_size("ns", "big_doc", large_text, {}, gzip_threshold_bytes=None)

    # With compression enabled (e.g. threshold 10 KB) -> passes safely
    check_item_size("ns", "big_doc", large_text, {}, gzip_threshold_bytes=10 * 1024)


def test_put_many_with_gzip_writes_compressed_payload():
    config = DynavecConfig(
        vector_bucket="b",
        index="i",
        table="docs",
        dimension=4,
        gzip_threshold_bytes=100,
    )
    session = MagicMock()
    table = session.resource.return_value.Table.return_value
    batch_context = table.batch_writer.return_value.__enter__.return_value

    store = DynamoDBStore(config, boto_session=session)

    long_text = "compress me! " * 50
    store.put_many("tenant", [("doc1", long_text, {"author": "Shivam"})])

    batch_context.put_item.assert_called_once()
    saved_item = batch_context.put_item.call_args[1]["Item"]
    assert "text_gzip" in saved_item
    assert "text" not in saved_item
    assert gzip.decompress(saved_item["text_gzip"]).decode("utf-8") == long_text
