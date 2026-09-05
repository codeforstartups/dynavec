"""Core data models used across dynavec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

Vector = list[float]
Metadata = dict[str, Any]


@dataclass
class Document:
    """A unit of content stored in dynavec.

    Either ``text`` (which will be embedded by the configured embedder) or a
    pre-computed ``vector`` must be provided at upsert time.
    """

    id: str
    text: Optional[str] = None
    vector: Optional[Vector] = None
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
    distance: Optional[float] = None
    text: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    vector: Optional[Vector] = None

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
