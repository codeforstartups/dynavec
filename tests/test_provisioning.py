import boto3
from moto import mock_aws

from dynavec.config import DynavecConfig
from dynavec.provisioning import ensure_table


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

        table = session.client("dynamodb").describe_table(
            TableName="test-table"
        )["Table"]

        assert table["KeySchema"] == [
            {"AttributeName": "pk", "KeyType": "HASH"}
        ]
        assert table["AttributeDefinitions"] == [
            {"AttributeName": "pk", "AttributeType": "S"}
        ]
