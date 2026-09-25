"""Exception hierarchy for dynavec."""

from __future__ import annotations


class DynavecError(Exception):
    """Base class for all dynavec errors."""


class ConfigurationError(DynavecError):
    """Raised when the client is misconfigured (bad region, missing table, etc.)."""


class ProvisioningError(DynavecError):
    """Raised when creating/verifying AWS resources fails."""


class EmbeddingError(DynavecError):
    """Raised when an embedder backend fails or is misconfigured."""


class DimensionMismatchError(DynavecError):
    """Raised when a vector's dimension does not match the index dimension."""


class NotFoundError(DynavecError):
    """Raised when a namespace, index, or document does not exist."""


class ItemTooLargeError(DynavecError):
    """Raised when a document would exceed DynamoDB's 400 KB item size limit."""

    def __init__(self, doc_id: str, namespace: str, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"Document {doc_id!r} in namespace {namespace!r} is about {size_bytes:,} bytes "
            f"as a DynamoDB item, over the {limit_bytes:,}-byte (400 KB) limit. Nothing "
            "was written. Split its text into smaller documents (e.g. with "
            "dynavec.ingest.chunk_text) or move large metadata values elsewhere."
        )
        self.doc_id = doc_id
        self.namespace = namespace
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


class ConflictError(DynavecError):
    """Raised when a document changed between being read and being written."""

    def __init__(self, doc_id: str, namespace: str, expected_version: int) -> None:
        super().__init__(
            f"Document {doc_id!r} in namespace {namespace!r} was modified concurrently: "
            f"expected version {expected_version}, but the stored version differs. "
            "Nothing was written. Re-read the document and retry the update."
        )
        self.doc_id = doc_id
        self.namespace = namespace
        self.expected_version = expected_version


class MissingDependencyError(DynavecError):
    """Raised when an optional dependency for a chosen backend is not installed."""

    def __init__(self, feature: str, package: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{package}' package. "
            f"Install it with:  pip install 'dynavec[{extra}]'"
        )
        self.feature = feature
        self.package = package
        self.extra = extra
