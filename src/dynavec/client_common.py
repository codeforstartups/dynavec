"""Pure helpers shared by sync and async Dynavec clients."""

from __future__ import annotations

import time
from typing import Any, Optional

from .config import DynavecConfig
from .exceptions import ConfigurationError, DimensionMismatchError
from .metadata import generate_auto_metadata, split_metadata
from .models import Document
from .stores.dynamodb import check_item_size
from .transforms import TransformContext, TransformPipeline
from .utils import KEY_SEPARATOR, decode_key_component, encode_key_component

Metadata = dict[str, Any]
S3Payload = tuple[str, list[float], Metadata]
DDBPayload = tuple[str, Optional[str], Metadata]
HotPayload = tuple[str, list[float], Optional[str], Metadata]
EmbeddingTarget = tuple[int, Optional[str]]


def s3_key(namespace: str, doc_id: str) -> str:
    return (
        f"{encode_key_component(namespace)}"
        f"{KEY_SEPARATOR}"
        f"{encode_key_component(doc_id)}"
    )


def split_key(key: str) -> tuple[str, str]:
    namespace, _, doc_id = key.partition(KEY_SEPARATOR)
    return (
        decode_key_component(namespace),
        decode_key_component(doc_id),
    )


def apply_transform_pipeline(
    docs: list[Document],
    namespace: str,
    pipeline: TransformPipeline | None,
) -> None:
    if pipeline is None:
        return

    for doc in docs:
        context = pipeline(
            TransformContext(
                id=doc.id,
                text=doc.text,
                vector=doc.vector,
                metadata=dict(doc.metadata),
                namespace=namespace,
            )
        )

        doc.text = context.text
        doc.vector = context.vector
        doc.metadata = context.metadata


def documents_to_embed(
    docs: list[Document],
) -> list[EmbeddingTarget]:
    return [
        (index, doc.text)
        for index, doc in enumerate(docs)
        if doc.vector is None
    ]


def embedding_texts(
    targets: list[EmbeddingTarget],
) -> list[str]:
    optional_texts = [text for _, text in targets]

    if any(text is None for text in optional_texts):
        raise ConfigurationError(
            "A document has neither text nor vector."
        )

    return [
        text
        for text in optional_texts
        if text is not None
    ]


def assign_embeddings(
    docs: list[Document],
    targets: list[EmbeddingTarget],
    vectors: list[list[float]],
) -> None:
    if len(vectors) != len(targets):
        raise ConfigurationError(
            f"Embedder returned {len(vectors)} vectors for "
            f"{len(targets)} documents."
        )

    for (index, _), vector in zip(targets, vectors):
        docs[index].vector = vector


def single_embedding(vectors: list[list[float]], doc_id: str) -> list[float]:
    if len(vectors) != 1:
        raise ConfigurationError(
            f"Embedder returned {len(vectors)} vectors for 1 document "
            f"({doc_id!r})."
        )

    return vectors[0]


def build_write_payloads(
    docs: list[Document],
    config: DynavecConfig,
    namespace: str,
    auto_metadata: bool,
    default_ttl_seconds: int | None = None,
) -> tuple[
    list[S3Payload],
    list[DDBPayload],
    list[str],
    list[HotPayload],
]:
    s3_payload: list[S3Payload] = []
    ddb_payload: list[DDBPayload] = []
    ids: list[str] = []
    hot_payload: list[HotPayload] = []

    for doc in docs:
        vector = doc.vector

        if vector is None:
            raise ConfigurationError(
                f"Embedder did not return a vector for document "
                f"{doc.id!r}."
            )

        if len(vector) != config.dimension:
            raise DimensionMismatchError(
                f"Document {doc.id!r} vector has dimension "
                f"{len(vector)}, expected {config.dimension}."
            )

        metadata = dict(doc.metadata)

        if auto_metadata:
            generated = generate_auto_metadata(doc.text)
            generated.update(metadata)
            metadata = generated

        s3_metadata, ddb_metadata = split_metadata(
            metadata,
            config,
            namespace,
            doc.text,
        )

        doc_ttl_seconds = (
            doc.ttl_seconds
            if doc.ttl_seconds is not None
            else default_ttl_seconds
        )

        if doc_ttl_seconds is not None and doc_ttl_seconds <= 0:
            raise ValueError(
                f"Document {doc.id!r} ttl_seconds must be positive, "
                f"got {doc_ttl_seconds}."
            )

        ttl_timestamp = (
            int(time.time() + doc_ttl_seconds)
            if doc_ttl_seconds is not None
            else None
        )

        if ttl_timestamp is not None:
            ddb_metadata["_ttl"] = ttl_timestamp

        check_item_size(
            namespace,
            doc.id,
            doc.text,
            ddb_metadata,
            config.gzip_threshold_bytes,
            ttl=ttl_timestamp,
            ttl_attribute=config.dynamodb_ttl_attribute,
        )

        s3_payload.append(
            (
                s3_key(namespace, doc.id),
                vector,
                s3_metadata,
            )
        )

        ddb_payload.append(
            (
                doc.id,
                doc.text,
                ddb_metadata,
            )
        )

        ids.append(doc.id)

        hot_payload.append(
            (
                doc.id,
                vector,
                doc.text,
                metadata,
            )
        )

    return s3_payload, ddb_payload, ids, hot_payload
