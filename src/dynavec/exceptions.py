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
