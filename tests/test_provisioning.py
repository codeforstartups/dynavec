from unittest.mock import MagicMock

import boto3
from moto import mock_aws

from dynavec.config import TEXT_METADATA_KEY, DynavecConfig
from dynavec.provisioning import ensure_index, ensure_table


def test_ensure_table_creates_expected_key_schema():
    with mock_aws():
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )

        config = DynavecConfig(
            vector_bucket="test-bucket",
            index="test-index",
            table="test-table",
            dimension=4,
            region="us-east-1",
        )

        ensure_table(config, session)

        table = session.client("dynamodb").describe_table(TableName="test-table")["Table"]

        assert table["KeySchema"] == [{"AttributeName": "pk", "KeyType": "HASH"}]
        assert table["AttributeDefinitions"] == [{"AttributeName": "pk", "AttributeType": "S"}]


def test_ensure_index_marks_text_mirror_non_filterable():
    s3vectors = MagicMock()

    session = MagicMock()
    session.client.return_value = s3vectors

    config = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
        store_text_in_s3vectors=True,
    )

    ensure_index(config, session)

    s3vectors.create_index.assert_called_once_with(
        vectorBucketName="test-bucket",
        indexName="test-index",
        dataType="float32",
        dimension=4,
        distanceMetric="cosine",
        metadataConfiguration={"nonFilterableMetadataKeys": [TEXT_METADATA_KEY]},
    )


def test_ensure_index_combines_configured_and_text_non_filterable_keys():
    s3vectors = MagicMock()

    session = MagicMock()
    session.client.return_value = s3vectors

    config = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="test-table",
        dimension=4,
        region="us-east-1",
        non_filterable_keys=["content", "summary"],
        store_text_in_s3vectors=True,
    )

    ensure_index(config, session)

    s3vectors.create_index.assert_called_once_with(
        vectorBucketName="test-bucket",
        indexName="test-index",
        dataType="float32",
        dimension=4,
        distanceMetric="cosine",
        metadataConfiguration={
            "nonFilterableMetadataKeys": [
                "content",
                "summary",
                TEXT_METADATA_KEY,
            ]
        },
    )
