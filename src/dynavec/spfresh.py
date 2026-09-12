"""SPFresh-style incremental dynamic partitioning for in-memory hot tier index.

Implements space partitioning with dynamic partition splitting (2-means),
dynamic partition merging (nearest centroid pairing), and centroid drift tracking.

References:
    SPFresh: Incremental In-Place Update for Billion-Scale Vector Search
    (Zhang et al., NeurIPS 2023 / arXiv:2310.13886)
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dynavec.metrics import Metric, score
from dynavec.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SPFreshConfig:
    """Configuration for SPFresh dynamic hot tier index."""

    dimension: int = 1536
    metric: Metric = "cosine"
    max_partition_size: int = 64
    min_partition_size: int = 16
    drift_threshold: float = 0.25
    n_probe: int = 3
    auto_rebalance: bool = True

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        if self.min_partition_size < 1:
            raise ValueError(f"min_partition_size must be >= 1, got {self.min_partition_size}")
        if self.max_partition_size <= self.min_partition_size:
            raise ValueError(
                f"max_partition_size ({self.max_partition_size}) must be greater than "
                f"min_partition_size ({self.min_partition_size})"
            )
        if self.drift_threshold <= 0:
            raise ValueError(f"drift_threshold must be positive, got {self.drift_threshold}")
        if self.n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {self.n_probe}")


class Partition:
    """A dynamic vector partition holding a cluster of vectors and its centroid.

    Thread-safety: All public methods acquire ``self.lock`` (RLock) internally.
    Callers that already hold the lock must use the internal ``_buf`` / ``_id_to_idx``
    attributes directly to avoid deadlock.
    """

    def __init__(
        self,
        partition_id: str,
        dimension: int,
        centroid: np.ndarray | None = None,
        metric: Metric = "cosine",
    ) -> None:
        self.id: str = partition_id
        self.dimension: int = dimension
        self.metric: Metric = metric
        self.lock: threading.RLock = threading.RLock()
        self.vector_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        # Pre-allocated contiguous buffer; doubles on capacity overflow
        self._capacity: int = 32
        self._buf: np.ndarray = np.empty((self._capacity, dimension), dtype=np.float32)
        self.metadata: dict[str, dict[str, Any]] = {}
        self.texts: dict[str, str | None] = {}
        self.version: int = 0

        if centroid is not None:
            c = np.asarray(centroid, dtype=np.float32).reshape(-1)
            if c.shape[0] != dimension:
                raise ValueError(
                    f"Centroid dimension mismatch: expected {dimension}, got {c.shape[0]}"
                )
            self.centroid: np.ndarray = self._normalize_if_needed(c)
            self.anchor_centroid: np.ndarray = self.centroid.copy()
        else:
            self.centroid = np.zeros(dimension, dtype=np.float32)
            self.anchor_centroid = np.zeros(dimension, dtype=np.float32)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _normalize_if_needed(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalise *vec* when using the cosine metric."""
        if self.metric == "cosine":
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                return (vec / norm).astype(np.float32)
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> np.ndarray:
        """View of currently stored vectors (not thread-safe; use snapshot() instead)."""
        return self._buf[: len(self.vector_ids)]

    def size(self) -> int:
        """Number of vectors currently in the partition."""
        with self.lock:
            return len(self.vector_ids)

    def add(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> None:
        """Add or update a vector in this partition."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {v.shape[0]}"
            )

        with self.lock:
            if vector_id in self._id_to_idx:
                # In-place update
                idx = self._id_to_idx[vector_id]
                self._buf[idx] = v
            else:
                n = len(self.vector_ids)
                if n >= self._capacity:
                    # Double the buffer
                    new_cap = max(self._capacity * 2, n + 1)
                    new_buf = np.empty((new_cap, self.dimension), dtype=np.float32)
                    if n > 0:
                        new_buf[:n] = self._buf[:n]
                    self._buf = new_buf
                    self._capacity = new_cap
                self._buf[n] = v
                self.vector_ids.append(vector_id)
                self._id_to_idx[vector_id] = n

            self.metadata[vector_id] = metadata or {}
            self.texts[vector_id] = text
            self.version += 1
            self.update_centroid(reanchor=False)

    def remove(self, vector_id: str) -> bool:
        """Remove a vector from this partition.

        Uses swap-with-last-element for O(1) removal.
        Returns True if removed, False if not found.
        """
        with self.lock:
            if vector_id not in self._id_to_idx:
                return False
            idx = self._id_to_idx[vector_id]
            last_idx = len(self.vector_ids) - 1

            if idx != last_idx:
                last_id = self.vector_ids[last_idx]
                self._buf[idx] = self._buf[last_idx]
                self.vector_ids[idx] = last_id
                self._id_to_idx[last_id] = idx

            self.vector_ids.pop()
            del self._id_to_idx[vector_id]
            self.metadata.pop(vector_id, None)
            self.texts.pop(vector_id, None)
            self.version += 1

            if self.vector_ids:
                self.update_centroid(reanchor=False)
            return True

    def update_centroid(self, reanchor: bool = False) -> None:
        """Recompute the centroid from all stored vectors (caller must hold lock)."""
        # NOTE: Called internally while lock is already held via add/remove;
        # RLock allows re-entrant acquisition so acquiring here is safe.
        with self.lock:
            n = len(self.vector_ids)
            if n == 0:
                return
            mean_c = np.mean(self._buf[:n], axis=0)
            self.centroid = self._normalize_if_needed(mean_c)
            # Set anchor on first population or when explicitly requested
            if reanchor or np.linalg.norm(self.anchor_centroid) < 1e-12:
                self.anchor_centroid = self.centroid.copy()

    def drift(self) -> float:
        """Normalised angular/Euclidean drift from the anchor centroid (0 = no drift)."""
        with self.lock:
            if not self.vector_ids:
                return 0.0
            diff = np.linalg.norm(self.centroid - self.anchor_centroid)
            base = np.linalg.norm(self.anchor_centroid)
            return float(diff / (base + 1e-12))

    def reset_anchor(self) -> None:
        """Anchor the current centroid as the new baseline for drift measurement."""
        with self.lock:
            self.anchor_centroid = self.centroid.copy()

    def snapshot(
        self,
    ) -> tuple[list[str], np.ndarray, dict[str, dict[str, Any]], dict[str, str | None]]:
        """Return an immutable copy of the partition state (thread-safe)."""
        with self.lock:
            n = len(self.vector_ids)
            return (
                list(self.vector_ids),
                self._buf[:n].copy() if n > 0 else np.empty((0, self.dimension), dtype=np.float32),
                dict(self.metadata),
                dict(self.texts),
            )


# ---------------------------------------------------------------------------
# Main Index
# ---------------------------------------------------------------------------


class SPFreshHotIndex:
    """Dynamic Space-Partitioned ANN index with incremental hot-tier rebalancing.

    Based on the SPFresh algorithm (Zhang et al., NeurIPS 2023):
    - Dynamic 2-means partition splitting when partitions overflow ``max_partition_size``.
    - Nearest-centroid partition merging when partitions drop below ``min_partition_size``.
    - Centroid drift tracking with boundary vector migration.
    - Fine-grained per-partition RLock for concurrent reads/writes.
    """

    def __init__(
        self,
        config: SPFreshConfig | None = None,
        dimension: int = 1536,
        metric: Metric = "cosine",
        max_partition_size: int = 64,
        min_partition_size: int = 16,
        drift_threshold: float = 0.25,
        n_probe: int = 3,
        auto_rebalance: bool = True,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SPFreshConfig(
                dimension=dimension,
                metric=metric,
                max_partition_size=max_partition_size,
                min_partition_size=min_partition_size,
                drift_threshold=drift_threshold,
                n_probe=n_probe,
                auto_rebalance=auto_rebalance,
            )

        # Global index lock (RLock allows re-entrant locking within same thread)
        self._lock = threading.RLock()
        self._partitions: dict[str, Partition] = {}
        self._id_to_partition: dict[str, str] = {}

    # ------------------------------------------------------------------ props

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def metric(self) -> Metric:
        return self.config.metric

    def __len__(self) -> int:
        with self._lock:
            return len(self._id_to_partition)

    @property
    def partition_count(self) -> int:
        with self._lock:
            return len(self._partitions)

    # ------------------------------------------------------------------ routing

    def _get_centroids_matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (partition_ids, centroids_matrix). Caller must hold ``_lock``."""
        pids = list(self._partitions.keys())
        if not pids:
            return [], np.empty((0, self.dimension), dtype=np.float32)
        mat = np.vstack([self._partitions[pid].centroid for pid in pids])
        return pids, mat

    def _find_nearest_partitions(self, query: np.ndarray, n_probe: int) -> list[tuple[str, float]]:
        """Top-``n_probe`` nearest partition centroids. Caller must hold ``_lock``."""
        pids, centroids = self._get_centroids_matrix()
        if not pids:
            return []
        scores = score(query, centroids, self.metric)
        k = min(n_probe, len(pids))
        top_indices = np.argsort(-scores)[:k]
        return [(pids[idx], float(scores[idx])) for idx in top_indices]

    # ------------------------------------------------------------------ write

    def insert(
        self,
        id: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> str:
        """Insert or update a vector. Returns the partition ID where it was stored.

        If the target partition exceeds ``max_partition_size`` and
        ``auto_rebalance`` is True, a split is triggered immediately and the
        returned partition ID reflects the child partition that actually holds
        the new vector.
        """
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {v.shape[0]}"
            )

        with self._lock:
            # Remove from old partition first (for upsert semantics)
            if id in self._id_to_partition:
                old_pid = self._id_to_partition[id]
                if old_pid in self._partitions:
                    self._partitions[old_pid].remove(id)
                del self._id_to_partition[id]

            if not self._partitions:
                pid = f"p-{uuid.uuid4().hex[:8]}"
                part = Partition(
                    partition_id=pid,
                    dimension=self.dimension,
                    centroid=v,
                    metric=self.metric,
                )
                self._partitions[pid] = part
                target_pid = pid
            else:
                target_pid = self._find_nearest_partitions(v, n_probe=1)[0][0]

            self._partitions[target_pid].add(vector_id=id, vector=v, metadata=metadata, text=text)
            self._id_to_partition[id] = target_pid

            # Auto-split if partition overflows
            if (
                self.config.auto_rebalance
                and self._partitions[target_pid].size() > self.config.max_partition_size
            ):
                result = self.split_partition(target_pid)
                if result:
                    # Update target_pid to the child that actually holds `id`
                    target_pid = self._id_to_partition.get(id, target_pid)

        return target_pid

    def insert_batch(
        self,
        ids: list[str],
        vectors: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str | None] | None = None,
    ) -> list[str]:
        """Insert a batch of vectors. Returns a list of partition IDs."""
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[0] != len(ids):
            raise ValueError(f"Length mismatch: {len(ids)} ids vs {vecs.shape[0]} vectors")

        pids = []
        for i, vid in enumerate(ids):
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            txt = texts[i] if texts and i < len(texts) else None
            pid = self.insert(id=vid, vector=vecs[i], metadata=meta, text=txt)
            pids.append(pid)
        return pids

    def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if deleted, False if not found."""
        with self._lock:
            if id not in self._id_to_partition:
                return False
            pid = self._id_to_partition.pop(id)
            partition = self._partitions.get(pid)
            if partition is None:
                return False

            removed = partition.remove(id)

            # Auto-merge if partition falls below minimum size
            if (
                self.config.auto_rebalance
                and len(self._partitions) > 1
                and partition.size() < self.config.min_partition_size
            ):
                self.merge_partition(pid)

            return removed

    def delete_batch(self, ids: list[str]) -> int:
        """Delete multiple vectors. Returns count of vectors successfully deleted."""
        return sum(1 for vid in ids if self.delete(vid))

    def get(self, id: str) -> tuple[np.ndarray, dict[str, Any], str | None] | None:
        """Retrieve a stored vector and its metadata by ID. Returns None if not found."""
        with self._lock:
            pid = self._id_to_partition.get(id)
            if not pid or pid not in self._partitions:
                return None
            part = self._partitions[pid]

        with part.lock:
            if id not in part._id_to_idx:
                return None
            idx = part._id_to_idx[id]
            return (
                part._buf[idx].copy(),
                dict(part.metadata.get(id, {})),
                part.texts.get(id),
            )

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: list[float] | np.ndarray,
        top_k: int = 10,
        nprobe: int | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[SearchResult]:
        """Approximate Nearest Neighbour search via dynamic space-partition routing.

        Args:
            query:     Query vector (must match configured dimension).
            top_k:     Maximum number of results to return.
            nprobe:    Number of partitions to probe. Defaults to ``config.n_probe``.
                       Higher values increase recall at the cost of latency.
            filter_fn: Optional callable that receives a vector's metadata dict and
                       returns False to exclude it from results.

        Returns:
            Up to ``top_k`` SearchResult objects sorted by descending similarity.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {q.shape[0]}"
            )

        probes = nprobe if nprobe is not None else self.config.n_probe

        # Snapshot the partition IDs to probe under the lock, then release
        with self._lock:
            if not self._partitions:
                return []
            active_pids = [
                pid
                for pid, _ in self._find_nearest_partitions(q, n_probe=probes)
                if pid in self._partitions
            ]

        hits: list[SearchResult] = []
        for pid in active_pids:
            # Grab a snapshot so searches are non-blocking
            with self._lock:
                part = self._partitions.get(pid)
                if part is None:
                    continue

            v_ids, v_mat, v_meta, v_text = part.snapshot()
            if not v_ids:
                continue

            scores_arr = score(q, v_mat, self.metric)
            for i, vid in enumerate(v_ids):
                meta = v_meta.get(vid, {})
                if filter_fn is not None and not filter_fn(meta):
                    continue
                hits.append(
                    SearchResult(
                        id=vid,
                        score=float(scores_arr[i]),
                        text=v_text.get(vid),
                        metadata=meta,
                        vector=v_mat[i].tolist(),
                    )
                )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------ rebalancing

    def split_partition(self, partition_id: str) -> tuple[str, str] | None:
        """Split an overloaded partition using spherical 2-means bipartitioning.

        Caller must hold ``_lock``.

        Returns:
            (child_pid_1, child_pid_2) on success, None if the partition cannot
            be split (missing, or has fewer than 2 vectors).
        """
        part = self._partitions.get(partition_id)
        if part is None or part.size() < 2:
            return None

        v_ids, v_mat, v_meta, v_text = part.snapshot()
        n = len(v_ids)
        if n < 2:
            return None

        # ---- Seed selection: farthest-first initialisation ----
        seed_idx_1 = 0
        if self.metric == "cosine":
            # Smallest cosine similarity = most orthogonal/opposite point
            sim = score(v_mat[seed_idx_1], v_mat, "cosine")
            seed_idx_2 = int(np.argmin(sim))
        else:
            dists = np.linalg.norm(v_mat - v_mat[seed_idx_1], axis=1)
            seed_idx_2 = int(np.argmax(dists))

        if seed_idx_2 == seed_idx_1:
            seed_idx_2 = (seed_idx_1 + 1) % n  # safe fallback

        c1 = v_mat[seed_idx_1].copy()
        c2 = v_mat[seed_idx_2].copy()

        # ---- Spherical 2-means ----
        assignments = np.zeros(n, dtype=np.int32)
        for _ in range(15):  # up to 15 iterations for tight convergence
            s1 = score(c1, v_mat, self.metric)
            s2 = score(c2, v_mat, self.metric)
            new_assignments = (s2 > s1).astype(np.int32)

            idx1 = np.where(new_assignments == 0)[0]
            idx2 = np.where(new_assignments == 1)[0]

            # Guard: if one cluster is empty, bisect and break
            if len(idx1) == 0 or len(idx2) == 0:
                mid = n // 2
                assignments[:mid] = 0
                assignments[mid:] = 1
                # Recompute centroids for the forced split
                idx1 = np.arange(mid)
                idx2 = np.arange(mid, n)
                c1 = np.mean(v_mat[idx1], axis=0)
                c2 = np.mean(v_mat[idx2], axis=0)
                break

            if np.array_equal(assignments, new_assignments):
                break  # converged

            assignments = new_assignments
            mean1 = np.mean(v_mat[idx1], axis=0)
            mean2 = np.mean(v_mat[idx2], axis=0)
            if self.metric == "cosine":
                n1, n2 = np.linalg.norm(mean1), np.linalg.norm(mean2)
                c1 = (mean1 / (n1 + 1e-12)).astype(np.float32)
                c2 = (mean2 / (n2 + 1e-12)).astype(np.float32)
            else:
                c1 = mean1.astype(np.float32)
                c2 = mean2.astype(np.float32)

        # ---- Build child partitions ----
        pid1 = f"{partition_id}-a-{uuid.uuid4().hex[:4]}"
        pid2 = f"{partition_id}-b-{uuid.uuid4().hex[:4]}"

        child1 = Partition(
            partition_id=pid1, dimension=self.dimension, centroid=c1, metric=self.metric
        )
        child2 = Partition(
            partition_id=pid2, dimension=self.dimension, centroid=c2, metric=self.metric
        )

        for i, vid in enumerate(v_ids):
            dest = child1 if assignments[i] == 0 else child2
            dest_pid = pid1 if assignments[i] == 0 else pid2
            dest.add(
                vector_id=vid,
                vector=v_mat[i],
                metadata=v_meta.get(vid),
                text=v_text.get(vid),
            )
            self._id_to_partition[vid] = dest_pid

        child1.reset_anchor()
        child2.reset_anchor()
        self._partitions[pid1] = child1
        self._partitions[pid2] = child2
        del self._partitions[partition_id]

        logger.info(
            "SPFresh split %s -> %s (%d) + %s (%d)",
            partition_id,
            pid1,
            child1.size(),
            pid2,
            child2.size(),
        )
        return pid1, pid2

    def merge_partition(self, partition_id: str) -> str | None:
        """Merge an undersized partition into its nearest neighbour.

        Caller must hold ``_lock``.

        The merge is skipped (returns None) if:
        - Only one partition exists.
        - The combined size would exceed ``max_partition_size`` (deferred merge).

        Returns the target partition ID on success, None otherwise.
        """
        part = self._partitions.get(partition_id)
        if part is None or len(self._partitions) <= 1:
            return None

        v_ids, v_mat, v_meta, v_text = part.snapshot()

        # Find nearest neighbour
        other_pids = [p for p in self._partitions if p != partition_id]
        other_centroids = np.vstack([self._partitions[p].centroid for p in other_pids])
        nbr_scores = score(part.centroid, other_centroids, self.metric)
        best_idx = int(np.argmax(nbr_scores))
        target_pid = other_pids[best_idx]
        target_part = self._partitions[target_pid]

        # Guard: skip merge if combined size would overflow max_partition_size
        if target_part.size() + len(v_ids) > self.config.max_partition_size:
            logger.debug(
                "SPFresh deferred merge %s -> %s: combined size %d > max %d",
                partition_id,
                target_pid,
                target_part.size() + len(v_ids),
                self.config.max_partition_size,
            )
            return None

        for i, vid in enumerate(v_ids):
            target_part.add(
                vector_id=vid,
                vector=v_mat[i],
                metadata=v_meta.get(vid),
                text=v_text.get(vid),
            )
            self._id_to_partition[vid] = target_pid

        target_part.update_centroid(reanchor=False)
        del self._partitions[partition_id]

        logger.info(
            "SPFresh merged %s (%d vectors) into %s (total: %d)",
            partition_id,
            len(v_ids),
            target_pid,
            target_part.size(),
        )
        return target_pid

    def check_drift(self) -> dict[str, float]:
        """Return a dict mapping every partition ID to its current centroid drift score."""
        with self._lock:
            return {pid: part.drift() for pid, part in self._partitions.items()}

    def rebalance(self) -> dict[str, Any]:
        """Execute one full incremental rebalance pass.

        Steps (all under global ``_lock``):
        1. Split any partition exceeding ``max_partition_size``.
        2. Merge any partition below ``min_partition_size`` (skip if merge would overflow).
        3. Reanchor centroids that have drifted beyond ``drift_threshold``.
        4. Migrate boundary vectors that belong more tightly to a neighbouring partition.

        Returns a summary dict with counts for observability.
        """
        splits: list[tuple[str, str]] = []
        merges: list[tuple[str, str]] = []
        drifts_reanchored: list[str] = []
        migrated_vectors: int = 0

        with self._lock:
            # ---- Phase 1: Split overloaded partitions ----
            for pid in list(self._partitions):
                part = self._partitions.get(pid)
                if part and part.size() > self.config.max_partition_size:
                    result = self.split_partition(pid)
                    if result:
                        splits.append(result)

            # ---- Phase 2: Merge underloaded partitions ----
            if len(self._partitions) > 1:
                for pid in list(self._partitions):
                    part = self._partitions.get(pid)
                    if part and part.size() < self.config.min_partition_size:
                        target = self.merge_partition(pid)
                        if target:
                            merges.append((pid, target))

            # ---- Phase 3: Drift detection & centroid reanchoring ----
            for pid, part in list(self._partitions.items()):
                if part.drift() >= self.config.drift_threshold:
                    part.reset_anchor()
                    drifts_reanchored.append(pid)

            # ---- Phase 4: Boundary vector migration ----
            # Take a fresh snapshot of partition IDs *after* splits/merges
            if len(self._partitions) > 1:
                for pid in list(self._partitions):
                    part = self._partitions.get(pid)
                    if not part or part.size() <= self.config.min_partition_size:
                        continue

                    v_ids, v_mat, v_meta, v_text = part.snapshot()
                    if not v_ids:
                        continue

                    other_pids = [p for p in self._partitions if p != pid]
                    if not other_pids:
                        continue
                    other_centroids = np.vstack([self._partitions[p].centroid for p in other_pids])

                    for i, vid in enumerate(v_ids):
                        # Skip vectors that no longer belong to this partition
                        # (may have been migrated in a previous iteration of this loop)
                        if self._id_to_partition.get(vid) != pid:
                            continue

                        vec = v_mat[i]
                        own_score = float(score(vec, part.centroid.reshape(1, -1), self.metric)[0])
                        other_scores = score(vec, other_centroids, self.metric)
                        best_other_idx = int(np.argmax(other_scores))
                        best_other_score = float(other_scores[best_other_idx])

                        # Migrate only with a clear margin (avoid churn on tie)
                        if best_other_score > own_score + 0.15:
                            target_pid = other_pids[best_other_idx]
                            target_part = self._partitions.get(target_pid)
                            if target_part and target_part.size() < self.config.max_partition_size:
                                part.remove(vid)
                                target_part.add(
                                    vector_id=vid,
                                    vector=vec,
                                    metadata=v_meta.get(vid),
                                    text=v_text.get(vid),
                                )
                                self._id_to_partition[vid] = target_pid
                                migrated_vectors += 1

        return {
            "splits": splits,
            "merges": merges,
            "drifts_reanchored": drifts_reanchored,
            "migrated_vectors": migrated_vectors,
            "total_partitions": self.partition_count,
            "total_vectors": len(self),
        }

    def stats(self) -> dict[str, Any]:
        """Return diagnostic health statistics of the index."""
        with self._lock:
            pids = list(self._partitions.keys())
            sizes = [self._partitions[p].size() for p in pids]
            drifts = [self._partitions[p].drift() for p in pids]

            return {
                "total_vectors": len(self._id_to_partition),
                "total_partitions": len(pids),
                "partition_sizes": {
                    "min": int(np.min(sizes)) if sizes else 0,
                    "max": int(np.max(sizes)) if sizes else 0,
                    "mean": float(np.mean(sizes)) if sizes else 0.0,
                    "std": float(np.std(sizes)) if sizes else 0.0,
                },
                "drift": {
                    "max": float(np.max(drifts)) if drifts else 0.0,
                    "mean": float(np.mean(drifts)) if drifts else 0.0,
                },
                "metric": self.metric,
                "dimension": self.dimension,
                "max_partition_size": self.config.max_partition_size,
                "min_partition_size": self.config.min_partition_size,
            }

    def clear(self) -> None:
        """Purge all partitions and vectors from the index."""
        with self._lock:
            self._partitions.clear()
            self._id_to_partition.clear()


# ---------------------------------------------------------------------------
# Background Rebalancer
# ---------------------------------------------------------------------------


class SPFreshRebalancer:
    """Background daemon thread that periodically calls ``SPFreshHotIndex.rebalance()``."""

    def __init__(self, index: SPFreshHotIndex, interval_seconds: float = 2.0) -> None:
        self.index = index
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background rebalancer thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="SPFreshRebalancerWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started SPFreshRebalancer (interval=%.2fs)", self.interval_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and block until it terminates."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("Stopped SPFreshRebalancer")

    def is_alive(self) -> bool:
        """Return True if the background worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict[str, Any]:
        """Execute a single rebalance pass synchronously (useful for testing)."""
        return self.index.rebalance()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.index.rebalance()
            except Exception as exc:  # pragma: no cover
                logger.error("SPFreshRebalancer error: %s", exc)
            self._stop_event.wait(timeout=self.interval_seconds)
