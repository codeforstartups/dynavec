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
class SearchExplanation:
    """Debug information collected for an explained search."""

    timings_ms: dict[str, float] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ExplainedSearchResult:
    """Search results together with per-stage debug information."""

    results: list[SearchResult] = field(default_factory=list)
    explanation: SearchExplanation = field(default_factory=SearchExplanation)


@dataclass
class IndexInfo:
    """Snapshot of the provisioned S3 Vectors index + DynamoDB table."""

    vector_bucket: str
    index: str
    dimension: int
    distance_metric: str
    table: str
    table_status: str
    non_filterable_keys: list[str] = field(default_factory=list)
    item_count: int | None = None


@dataclass
class UpsertResult:
    """Summary returned from an upsert call."""

    count: int
    ids: list[str] = field(default_factory=list)
    # Stored version after an update(); None for plain upserts.
    version: int | None = None
