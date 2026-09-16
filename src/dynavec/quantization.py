"""Product Quantization (PQ) for compact vector caching.

Where this helps: S3 Vectors stores float32 and manages its own layout, so PQ
does **not** change what's in S3 Vectors. PQ compresses the vectors dynavec
caches itself — the in-memory hot tier (v0.2) and any local candidate cache —
turning a ``dim × 4`` byte vector into ``m`` bytes (8-bit codes). For 768-dim,
``m=96`` gives a 32× memory reduction with a small, tunable accuracy cost.

Pure numpy, no external deps. Symmetric + asymmetric (ADC) distance supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

FORMAT_VERSION = 1


def _kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """Tiny Lloyd's k-means; returns (k, d) centroids."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n <= k:
        # pad by repeating points if we have fewer samples than clusters
        reps = int(np.ceil(k / n))
        x = np.tile(x, (reps, 1))[:k]
        return x.astype(np.float32)

    centroids = x[rng.choice(n, size=k, replace=False)].astype(np.float32)
    for _ in range(iters):
        # assign
        d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        # update
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return centroids


@dataclass
class ProductQuantizer:
    """Split each vector into ``m`` subvectors and quantize each independently.

    Parameters
    ----------
    m:
        Number of subspaces. Must divide the vector dimension.
    nbits:
        Bits per subquantizer code (8 -> 256 centroids/subspace, uint8 codes).
    iters, seed:
        k-means training controls.
    """

    m: int
    nbits: int = 8
    iters: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        self.ksub = 2**self.nbits
        self._codebooks: np.ndarray | None = None  # (m, ksub, dsub)
        self._dsub: int | None = None

    # ------------------------------------------------------------------ train
    def fit(self, vectors: np.ndarray) -> ProductQuantizer:
        x = np.asarray(vectors, dtype=np.float32)
        dim = x.shape[1]
        if dim % self.m != 0:
            raise ValueError(f"m={self.m} must divide dimension={dim}")
        self._dsub = dim // self.m
        books = np.zeros((self.m, self.ksub, self._dsub), dtype=np.float32)
        for j in range(self.m):
            sub = x[:, j * self._dsub : (j + 1) * self._dsub]
            books[j] = _kmeans(sub, self.ksub, self.iters, self.seed + j)
        self._codebooks = books
        return self

    @property
    def is_fitted(self) -> bool:
        return self._codebooks is not None

    @property
    def code_size_bytes(self) -> int:
        """Bytes per encoded vector (for 8-bit codes, == m)."""
        return self.m * (self.nbits // 8 or 1)

    # ----------------------------------------------------------------- encode
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        codes = np.empty((x.shape[0], self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = x[:, j * self._dsub : (j + 1) * self._dsub]
            d = ((sub[:, None, :] - self._codebooks[j][None, :, :]) ** 2).sum(axis=2)
            codes[:, j] = d.argmin(axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximate reconstruction from codes."""
        self._check_fitted()
        codes = np.atleast_2d(codes)
        out = np.empty((codes.shape[0], self.m * self._dsub), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self._dsub : (j + 1) * self._dsub] = self._codebooks[j][codes[:, j]]
        return out

    # --------------------------------------------------------------- distance
    def asymmetric_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """ADC: squared-L2 from a full-precision query to encoded vectors.

        Precomputes a per-subspace distance table so scoring N codes is a few
        table lookups — the reason PQ is fast at scale.
        """
        self._check_fitted()
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        codes = np.atleast_2d(codes)
        # distance table: (m, ksub)
        table = np.empty((self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            qsub = q[j * self._dsub : (j + 1) * self._dsub]
            table[j] = ((self._codebooks[j] - qsub) ** 2).sum(axis=1)
        # sum table lookups across subspaces
        dists = np.zeros(codes.shape[0], dtype=np.float32)
        for j in range(self.m):
            dists += table[j][codes[:, j]]
        return dists

    def reconstruction_error(self, vectors: np.ndarray) -> float:
        """Mean squared reconstruction error (quality diagnostic)."""
        x = np.asarray(vectors, dtype=np.float32)
        recon = self.decode(self.encode(x))
        return float(((x - recon) ** 2).sum(axis=1).mean())

    # ----------------------------------------------------------- serialization
    def save(self, file: str | Path | BinaryIO) -> None:
        """Serialize the trained ProductQuantizer to a file or file-like object.

        Includes a version header (``format_version=1``), configuration parameters
        (``m``, ``nbits``, ``iters``, ``seed``, ``dsub``), and the trained codebooks.

        Raises
        ------
        RuntimeError
            If the quantizer has not been .fit() yet.
        """
        self._check_fitted()
        payload = {
            "format_version": np.array(FORMAT_VERSION, dtype=np.int32),
            "m": np.array(self.m, dtype=np.int32),
            "nbits": np.array(self.nbits, dtype=np.int32),
            "iters": np.array(self.iters, dtype=np.int32),
            "seed": np.array(self.seed, dtype=np.int32),
            "dsub": np.array(self._dsub, dtype=np.int32),
            "codebooks": self._codebooks,
        }
        if isinstance(file, (str, Path)):
            with open(file, "wb") as f:
                np.savez(f, **payload)
        else:
            np.savez(file, **payload)

    @classmethod
    def load(cls, file: str | Path | BinaryIO) -> ProductQuantizer:
        """Load a serialized ProductQuantizer from a file or file-like object.

        Parameters
        ----------
        file:
            A file path or binary file-like object.

        Returns
        -------
        ProductQuantizer
            A fully fitted ProductQuantizer instance.

        Raises
        ------
        ValueError
            If the file format is invalid or has an unsupported version header.
        """
        try:
            with np.load(file, allow_pickle=False) as data:
                if "format_version" not in data:
                    raise ValueError("Invalid quantizer file: missing format_version header")
                version = int(data["format_version"])
                if version != FORMAT_VERSION:
                    raise ValueError(f"Unsupported quantizer format version: {version}")

                for key in ("m", "nbits", "iters", "seed", "dsub", "codebooks"):
                    if key not in data:
                        raise ValueError(f"Invalid quantizer file: missing '{key}' field")

                m = int(data["m"])
                nbits = int(data["nbits"])
                iters = int(data["iters"])
                seed = int(data["seed"])

                pq = cls(m=m, nbits=nbits, iters=iters, seed=seed)
                pq._dsub = int(data["dsub"])
                pq._codebooks = np.asarray(data["codebooks"], dtype=np.float32)
                return pq
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load ProductQuantizer: {exc}") from exc

    def _check_fitted(self) -> None:
        if self._codebooks is None:
            raise RuntimeError("ProductQuantizer must be .fit() before use")



@dataclass
class ScalarQuantizer:
    """Per-dimension INT8 scalar quantization."""

    def __post_init__(self):
        self._mins = None
        self._scales = None

    @property
    def is_fitted(self):
        return self._mins is not None

    @property
    def code_size_bytes(self):
        if self._mins is None:
            raise RuntimeError("ScalarQuantizer is not fitted")
        return self._mins.shape[0]

    def fit(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)

        scales = (maxs - mins) / 255.0
        scales = np.where(scales == 0, 1.0, scales)

        self._mins = mins
        self._scales = scales

        return self

    def encode(self, vectors):
        self._check_fitted()

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        codes = np.round(
            (vectors - self._mins) / self._scales - 128
        )

        return np.clip(codes, -128, 127).astype(np.int8)

    def decode(self, codes):
        self._check_fitted()

        codes = np.asarray(codes, dtype=np.int8)

        return (
            (codes.astype(np.float32) + 128) * self._scales
            + self._mins
        ).astype(np.float32)

    def reconstruction_error(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        reconstructed = self.decode(self.encode(vectors))

        return float(np.mean((vectors - reconstructed) ** 2))

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer must be .fit() before use")
