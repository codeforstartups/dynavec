"""Core data models used across dynavec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Vector = list[float]
Metadata = dict[str, Any]


@dataclass
class Document:
    """A unit of content stored in dynavec.

    Either ``text`` (which will be embedded by the configured embedder) or a
    pre-computed ``vector`` must be provided at upsert time.
    """

    id: str
    text: str | None = None
    vector: Vector | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text is None and self.vector is None:
            raise ValueError(
                f"Document {self.id!r} must have either 'text' or 'vector' set."
            )


@dataclass
class SearchResult:
    """A single hit returned from a query, hydrated from DynamoDB.

    ``score`` is normalized so that **higher is more similar** regardless of the
    index distance metric. ``distance`` is the raw value returned by S3 Vectors.
    """

    id: str
    score: float
    distance: float | None = None
    text: str | None = None
    metadata: Metadata = field(default_factory=dict)
    vector: Vector | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class UpsertResult:
    """Summary returned from an upsert call."""

    count: int
    ids: list[str] = field(default_factory=list)
    
@dataclass
class NamespaceStats:
    """Summary statistics for a namespace."""

    namespace: str
    document_count: int
    text_bytes: int = 0

    @property
    def approximate_size_bytes(self) -> int:
        """Approximate size of stored document text."""
        return self.text_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "document_count": self.document_count,
            "approximate_size_bytes": self.approximate_size_bytes,
        }
