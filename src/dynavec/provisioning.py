"""One-shot provisioning of the AWS resources dynavec needs.

Creates (idempotently) inside the *caller's own* account & region:
  * an S3 vector bucket
  * a vector index (dimension + distance metric)
  * a DynamoDB table (pk = "{namespace}#{id}", on-demand billing by default)

Everything is safe to call repeatedly; existing resources are left as-is.
"""

from __future__ import annotations

from typing import Any, cast

from .config import TEXT_METADATA_KEY, DynavecConfig
from .exceptions import ProvisioningError


def _client_error_code(exc: Exception) -> str:
    response = cast(dict[str, Any], getattr(exc, "response", {}))
    return str(response.get("Error", {}).get("Code", ""))


def ensure_vector_bucket(config: DynavecConfig, boto_session: Any | None = None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    client_kwargs: dict[str, object] = {"region_name": config.region}
    botocore_config = config.botocore_config()
    if botocore_config is not None:
        client_kwargs["config"] = botocore_config
    s3v = session.client("s3vectors", **client_kwargs)
    try:
        s3v.create_vector_bucket(vectorBucketName=config.vector_bucket)
    except Exception as exc:  # noqa: BLE001
        if _client_error_code(exc) in ("ConflictException", "BucketAlreadyOwnedByYou"):
            return
        raise ProvisioningError(f"Failed to create vector bucket: {exc}") from exc


def ensure_index(config: DynavecConfig, boto_session: Any | None = None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    client_kwargs: dict[str, object] = {"region_name": config.region}
    botocore_config = config.botocore_config()
    if botocore_config is not None:
        client_kwargs["config"] = botocore_config
    s3v = session.client("s3vectors", **client_kwargs)

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


def ensure_ttl(
    config: DynavecConfig,
    boto_session: Any | None = None,
    ttl_attribute: str | None = None,
) -> None:
    """Enable Time to Live (TTL) on the DynamoDB table. Idempotent."""
    import boto3

    session = boto_session or boto3.Session()
    client_kwargs: dict[str, object] = {"region_name": config.region}
    botocore_config = config.botocore_config()
    if botocore_config is not None:
        client_kwargs["config"] = botocore_config
    ddb = session.client("dynamodb", **client_kwargs)

    attr = ttl_attribute or config.dynamodb_ttl_attribute
    try:
        desc = ddb.describe_time_to_live(TableName=config.table)
        ttl_desc = desc.get("TimeToLiveDescription", {})
        status = ttl_desc.get("TimeToLiveStatus")
        current_attr = ttl_desc.get("AttributeName")
        if status in ("ENABLED", "ENABLING") and current_attr == attr:
            return
        ddb.update_time_to_live(
            TableName=config.table,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": attr,
            },
        )
    except Exception as exc:  # noqa: BLE001
        code = _client_error_code(exc)
        if code in ("ValidationException", "ResourceInUseException"):
            return
        raise ProvisioningError(
            f"Failed to enable TTL on DynamoDB table {config.table!r}: {exc}"
        ) from exc


def ensure_table(config: DynavecConfig, boto_session: Any | None = None) -> None:
    import boto3

    session = boto_session or boto3.Session()
    client_kwargs: dict[str, object] = {"region_name": config.region}
    botocore_config = config.botocore_config()
    if botocore_config is not None:
        client_kwargs["config"] = botocore_config
    ddb = session.client("dynamodb", **client_kwargs)

    table_created = False
    try:
        create_kwargs: dict[str, Any] = {
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
        table_created = True
    except Exception as exc:  # noqa: BLE001
        if _client_error_code(exc) != "ResourceInUseException":
            raise ProvisioningError(f"Failed to create DynamoDB table: {exc}") from exc

    # Wait until the table is ACTIVE before returning.
    if table_created:
        ddb.get_waiter("table_exists").wait(TableName=config.table)

    if config.dynamodb_enable_ttl:
        ensure_ttl(config, boto_session)


def provision_all(config: DynavecConfig, boto_session: Any | None = None) -> None:
    """Create every resource dynavec needs. Idempotent."""
    ensure_vector_bucket(config, boto_session)
    ensure_index(config, boto_session)
    ensure_table(config, boto_session)
