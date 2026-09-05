"""One-shot provisioning of the AWS resources dynavec needs.

Creates (idempotently) inside the *caller's own* account & region:
  * an S3 vector bucket
  * a vector index (dimension + distance metric)
  * a DynamoDB table (pk = "{namespace}#{id}", on-demand billing by default)

Everything is safe to call repeatedly; existing resources are left as-is.
"""

from __future__ import annotations

from .config import TEXT_METADATA_KEY, DynavecConfig
from .exceptions import ProvisioningError


def _client_error_code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


def ensure_vector_bucket(config: DynavecConfig, boto_session=None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    s3v = session.client("s3vectors", region_name=config.region)
    try:
        s3v.create_vector_bucket(vectorBucketName=config.vector_bucket)
    except Exception as exc:  # noqa: BLE001
        if _client_error_code(exc) in ("ConflictException", "BucketAlreadyOwnedByYou"):
            return
        raise ProvisioningError(f"Failed to create vector bucket: {exc}") from exc


def ensure_index(config: DynavecConfig, boto_session=None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    s3v = session.client("s3vectors", region_name=config.region)

    non_filterable = list(config.non_filterable_keys)
    if config.store_text_in_s3vectors and TEXT_METADATA_KEY not in non_filterable:
        non_filterable.append(TEXT_METADATA_KEY)

    kwargs = {
        "vectorBucketName": config.vector_bucket,
        "indexName": config.index,
        "dataType": "float32",
        "dimension": config.dimension,
        "distanceMetric": config.distance_metric,
    }
    if non_filterable:
        kwargs["metadataConfiguration"] = {"nonFilterableMetadataKeys": non_filterable}

    try:
        s3v.create_index(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _client_error_code(exc) == "ConflictException":
            return
        raise ProvisioningError(f"Failed to create vector index: {exc}") from exc


def ensure_table(config: DynavecConfig, boto_session=None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    ddb = session.client("dynamodb", region_name=config.region)

    try:
        create_kwargs = {
            "TableName": config.table,
            "AttributeDefinitions": [{"AttributeName": "pk", "AttributeType": "S"}],
            "KeySchema": [{"AttributeName": "pk", "KeyType": "HASH"}],
            "BillingMode": config.dynamodb_billing_mode,
        }
        if config.dynamodb_billing_mode == "PROVISIONED":
            create_kwargs["ProvisionedThroughput"] = {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            }
        ddb.create_table(**create_kwargs)
    except Exception as exc:  # noqa: BLE001
        if _client_error_code(exc) == "ResourceInUseException":
            return  # already exists
        raise ProvisioningError(f"Failed to create DynamoDB table: {exc}") from exc

    # Wait until the table is ACTIVE before returning.
    ddb.get_waiter("table_exists").wait(TableName=config.table)


def provision_all(config: DynavecConfig, boto_session=None) -> None:
    """Create every resource dynavec needs. Idempotent."""
    ensure_vector_bucket(config, boto_session)
    ensure_index(config, boto_session)
    ensure_table(config, boto_session)
