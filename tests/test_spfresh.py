"""Comprehensive and rigorous unit, stress, and concurrency tests for SPFresh."""

import threading
import time

import numpy as np
import pytest

from dynavec.metrics import score
from dynavec.spfresh import (
    Partition,
    SPFreshConfig,
    SPFreshHotIndex,
    SPFreshRebalancer,
)


def _random_normalized_vectors(n: int, d: int = 16) -> np.ndarray:
    """Generate n random L2-normalized vectors of dimension d."""
    vecs = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


class TestSPFreshConfig:
    def test_valid_config(self) -> None:
        cfg = SPFreshConfig(dimension=128, max_partition_size=50, min_partition_size=10)
        assert cfg.dimension == 128
        assert cfg.max_partition_size == 50
        assert cfg.min_partition_size == 10
        assert cfg.metric == "cosine"

    def test_invalid_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            SPFreshConfig(dimension=0)

    def test_invalid_partition_bounds(self) -> None:
        with pytest.raises(ValueError, match="must be greater than"):
            SPFreshConfig(max_partition_size=10, min_partition_size=20)

    def test_invalid_min_partition(self) -> None:
        with pytest.raises(ValueError, match="min_partition_size must be >= 1"):
            SPFreshConfig(min_partition_size=0, max_partition_size=10)

    def test_invalid_drift_threshold(self) -> None:
        with pytest.raises(ValueError, match="drift_threshold must be positive"):
            SPFreshConfig(drift_threshold=0.0)


class TestPartition:
    def test_partition_crud_and_preallocation(self) -> None:
        part = Partition(partition_id="p1", dimension=4)
        assert part.size() == 0

        # Test preallocated buffer growth beyond initial 32 capacity
        for i in range(50):
            v = np.array([float(i), 1.0, 0.0, 0.0], dtype=np.float32)
            part.add(f"doc-{i}", v, metadata={"idx": i})

        assert part.size() == 50
        assert len(part.vector_ids) == 50
        assert part.vectors.shape == (50, 4)

        # Update existing
        v_updated = np.array([999.0, 0.0, 0.0, 0.0], dtype=np.float32)
        part.add("doc-10", v_updated, metadata={"idx": "updated"})
        assert part.size() == 50

        # Remove elements
        assert part.remove("doc-10") is True
        assert part.size() == 49
        assert part.remove("non-existent") is False
        assert part.size() == 49

    def test_partition_drift_and_reset(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        part = Partition(partition_id="p1", dimension=4, centroid=v1)
        assert part.drift() == 0.0

        v_drift = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
        part.add("doc-drift", v_drift)

        drift = part.drift()
        assert drift > 0.0

        part.reset_anchor()
        assert part.drift() == 0.0


class TestSPFreshHotIndex:
    def test_insert_and_get(self) -> None:
        dim = 8
        index = SPFreshHotIndex(dimension=dim, max_partition_size=10, min_partition_size=2)
        vec = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

        pid = index.insert("doc-1", vec, metadata={"category": "ai"}, text="hello world")
        assert len(index) == 1
        assert pid is not None

        retrieved = index.get("doc-1")
        assert retrieved is not None
        v_out, meta_out, text_out = retrieved
        np.testing.assert_allclose(v_out, vec, atol=1e-5)
        assert meta_out == {"category": "ai"}
        assert text_out == "hello world"

        assert index.get("non-existent") is None

    def test_batch_operations(self) -> None:
        dim = 8
        index = SPFreshHotIndex(dimension=dim, max_partition_size=20, min_partition_size=2)
        vecs = _random_normalized_vectors(15, dim)
        ids = [f"id-{i}" for i in range(15)]
        metas = [{"num": i} for i in range(15)]

        pids = index.insert_batch(ids, vecs, metadatas=metas)
        assert len(pids) == 15
        assert len(index) == 15

        deleted = index.delete_batch(["id-0", "id-1", "id-999"])
        assert deleted == 2
        assert len(index) == 13

    def test_search_and_filter(self) -> None:
        dim = 4
        index = SPFreshHotIndex(
            dimension=dim, metric="cosine", max_partition_size=10, min_partition_size=2
        )

        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        index.insert("target-1", v1, metadata={"category": "tech"})
        index.insert("target-2", v2, metadata={"category": "sports"})
        index.insert("target-3", v3, metadata={"category": "tech"})

        results = index.search(v1, top_k=2)
        assert len(results) == 2
        assert results[0].id == "target-1"
        assert results[0].score > 0.99
        assert results[1].id == "target-2"

        filtered = index.search(
            v1,
            top_k=2,
            filter_fn=lambda m: m.get("category") == "tech",
        )
        assert len(filtered) == 2
        assert filtered[0].id == "target-1"
        assert filtered[1].id == "target-3"

    def test_dynamic_split_on_overflow(self) -> None:
        dim = 8
        max_size = 10
        min_size = 2
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=max_size,
            min_partition_size=min_size,
            auto_rebalance=True,
        )

        assert index.partition_count == 0

        vecs = _random_normalized_vectors(15, dim)
        for i in range(15):
            index.insert(f"doc-{i}", vecs[i])

        assert len(index) == 15
        assert index.partition_count >= 2

        for i in range(15):
            res = index.get(f"doc-{i}")
            assert res is not None

    def test_dynamic_merge_on_underflow(self) -> None:
        dim = 8
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=6,
            min_partition_size=3,
            auto_rebalance=True,
        )

        cluster_a = np.random.randn(5, dim).astype(np.float32) + 10.0
        cluster_b = np.random.randn(5, dim).astype(np.float32) - 10.0

        for i in range(5):
            index.insert(f"a-{i}", cluster_a[i])
            index.insert(f"b-{i}", cluster_b[i])

        assert len(index) == 10
        assert index.partition_count >= 2

        for i in range(4):
            index.delete(f"a-{i}")

        assert len(index) == 6
        assert index.get("a-4") is not None
        for i in range(5):
            assert index.get(f"b-{i}") is not None

    def test_manual_rebalance_and_drift(self) -> None:
        dim = 4
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=10,
            min_partition_size=2,
            drift_threshold=0.1,
            auto_rebalance=False,
        )

        vecs = _random_normalized_vectors(15, dim)
        for i in range(15):
            index.insert(f"item-{i}", vecs[i])

        assert index.partition_count == 1

        summary = index.rebalance()
        assert len(summary["splits"]) > 0
        assert index.partition_count >= 2

        drifts = index.check_drift()
        assert len(drifts) == index.partition_count

    def test_stats_reporting(self) -> None:
        dim = 4
        index = SPFreshHotIndex(dimension=dim, max_partition_size=10, min_partition_size=2)
        vecs = _random_normalized_vectors(5, dim)
        for i in range(5):
            index.insert(f"v-{i}", vecs[i])

        stats = index.stats()
        assert stats["total_vectors"] == 5
        assert stats["total_partitions"] >= 1
        assert "partition_sizes" in stats
        assert stats["dimension"] == dim

        index.clear()
        assert len(index) == 0
        assert index.partition_count == 0


class TestHardScenariosAndStress:
    def test_high_dimension_1536_openai(self) -> None:
        """Test with realistic 1536-dimensional OpenAI embeddings."""
        dim = 1536
        index = SPFreshHotIndex(
            dimension=dim,
            metric="cosine",
            max_partition_size=32,
            min_partition_size=8,
            auto_rebalance=True,
        )

        n = 100
        vecs = _random_normalized_vectors(n, dim)
        ids = [f"emb-{i}" for i in range(n)]

        index.insert_batch(ids, vecs)
        assert len(index) == n
        assert index.partition_count >= 3

        # Search query
        q = vecs[0]
        results = index.search(q, top_k=5, nprobe=index.partition_count)
        assert len(results) == 5
        assert results[0].id == "emb-0"
        assert results[0].score > 0.999

    def test_identical_and_collinear_vectors_split(self) -> None:
        """Verify 2-means split bisects gracefully when all vectors are identical."""
        dim = 4
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=4,
            min_partition_size=2,
            auto_rebalance=True,
        )

        identical_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(6):
            index.insert(f"dup-{i}", identical_vec)

        assert len(index) == 6
        assert index.partition_count >= 2
        for i in range(6):
            assert index.get(f"dup-{i}") is not None

    def test_edge_cases_empty_and_oversized_probes(self) -> None:
        """Test empty index search, oversized top_k, and oversized nprobe."""
        dim = 4
        index = SPFreshHotIndex(dimension=dim)

        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert index.search(q, top_k=10) == []
        assert index.delete("non-existent") is False

        # Insert 3 vectors
        index.insert("a", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        index.insert("b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        index.insert("c", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

        # Request top_k=50 with nprobe=100 on 3-item index
        results = index.search(q, top_k=50, nprobe=100)
        assert len(results) == 3
        assert results[0].id == "a"

    def test_recall_accuracy_vs_brute_force(self) -> None:
        """Measure Top-5 ANN recall of SPFresh against exhaustive brute force."""
        dim = 16
        n_clusters = 5
        vecs_per_cluster = 30

        # Generate clustered vectors around distinct centroids
        cluster_centers = _random_normalized_vectors(n_clusters, dim)
        all_vecs = []
        ids = []
        count = 0
        for k in range(n_clusters):
            center = cluster_centers[k]
            noise = np.random.randn(vecs_per_cluster, dim).astype(np.float32) * 0.1
            cluster_vecs = center + noise
            norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-12
            cluster_vecs = cluster_vecs / norms
            for v in cluster_vecs:
                all_vecs.append(v)
                ids.append(f"v-{count}")
                count += 1

        vecs = np.array(all_vecs, dtype=np.float32)

        index = SPFreshHotIndex(
            dimension=dim,
            metric="cosine",
            max_partition_size=20,
            min_partition_size=5,
            n_probe=3,
            auto_rebalance=True,
        )
        index.insert_batch(ids, vecs)

        # Run 10 query benchmarks from clustered points
        recalls = []
        for _ in range(10):
            # Query near one of the clusters
            center_idx = np.random.randint(0, n_clusters)
            q = cluster_centers[center_idx] + np.random.randn(dim).astype(np.float32) * 0.05
            q = q / (np.linalg.norm(q) + 1e-12)

            # Exact brute force Top-5
            exact_scores = score(q, vecs, "cosine")
            exact_top5_indices = set(np.argsort(-exact_scores)[:5])
            exact_top5_ids = {ids[idx] for idx in exact_top5_indices}

            ann_results = index.search(q, top_k=5, nprobe=3)
            ann_top5_ids = {r.id for r in ann_results}

            overlap = len(exact_top5_ids.intersection(ann_top5_ids))
            recalls.append(overlap / 5.0)

        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.85, f"Mean recall {mean_recall} is below threshold 0.85"

    def test_euclidean_and_dot_metrics(self) -> None:
        """Test SPFresh index under Euclidean and Dot product distance metrics."""
        dim = 8
        for m in ("euclidean", "dot"):
            index = SPFreshHotIndex(
                dimension=dim,
                metric=m,
                max_partition_size=6,
                min_partition_size=2,
                auto_rebalance=True,
            )
            vecs = _random_normalized_vectors(10, dim)
            for i in range(10):
                index.insert(f"m-{i}", vecs[i])

            assert len(index) == 10
            hits = index.search(vecs[0], top_k=3)
            assert len(hits) == 3
            assert hits[0].id == "m-0"

    def test_boundary_vector_migration_on_drift(self) -> None:
        """Test vector migration when centroids move closer to neighbor items."""
        dim = 4
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=20,
            min_partition_size=2,
            auto_rebalance=False,
        )

        # Manually create 2 distinct clusters
        c1_vecs = np.array(
            [[10.0, 0.0, 0.0, 0.0], [10.1, 0.1, 0.0, 0.0], [9.9, -0.1, 0.0, 0.0]], dtype=np.float32
        )
        c2_vecs = np.array(
            [[-10.0, 0.0, 0.0, 0.0], [-10.1, 0.1, 0.0, 0.0], [-9.9, -0.1, 0.0, 0.0]],
            dtype=np.float32,
        )

        for i, v in enumerate(c1_vecs):
            index.insert(f"c1-{i}", v)
        index.split_partition(list(index._partitions.keys())[0])

        for i, v in enumerate(c2_vecs):
            index.insert(f"c2-{i}", v)

        # Add an item placed near cluster 2 into cluster 1's partition
        pids = list(index._partitions.keys())
        p_c1 = pids[0]
        v_outlier = np.array([-9.95, 0.05, 0.0, 0.0], dtype=np.float32)
        index._partitions[p_c1].add("outlier-1", v_outlier)
        index._id_to_partition["outlier-1"] = p_c1

        # Rebalance should detect and migrate the outlier vector to the closer cluster
        summary = index.rebalance()
        assert summary["total_vectors"] == len(index)


class TestSPFreshRebalancerWorker:
    def test_background_worker_lifecycle(self) -> None:
        dim = 8
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=8,
            min_partition_size=2,
            auto_rebalance=False,
        )
        rebalancer = SPFreshRebalancer(index=index, interval_seconds=0.1)

        vecs = _random_normalized_vectors(12, dim)
        for i in range(12):
            index.insert(f"doc-{i}", vecs[i])

        assert index.partition_count == 1

        res = rebalancer.run_once()
        assert len(res["splits"]) >= 1

        assert not rebalancer.is_alive()
        rebalancer.start()
        assert rebalancer.is_alive()
        time.sleep(0.2)
        rebalancer.stop()
        assert not rebalancer.is_alive()


class TestConcurrency:
    def test_heavy_concurrent_reads_writes_splits_and_rebalances(self) -> None:
        """Stress test with 8 concurrent threads under continuous splits, deletes, queries, and background rebalance."""
        dim = 16
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=8,
            min_partition_size=2,
            auto_rebalance=True,
        )
        rebalancer = SPFreshRebalancer(index=index, interval_seconds=0.05)
        rebalancer.start()

        errors: list[Exception] = []
        stop_event = threading.Event()

        def writer(worker_id: int) -> None:
            try:
                for i in range(40):
                    v = np.random.randn(dim).astype(np.float32)
                    doc_id = f"w-{worker_id}-{i}"
                    index.insert(doc_id, v, metadata={"worker": worker_id, "seq": i})
                    if i % 4 == 0:
                        index.delete(f"w-{worker_id}-{i - 2}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader(reader_id: int) -> None:
            try:
                q = np.random.randn(dim).astype(np.float32)
                while not stop_event.is_set():
                    hits = index.search(q, top_k=5, nprobe=3)
                    assert isinstance(hits, list)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        # Launch 4 writer threads
        for w_id in range(4):
            t = threading.Thread(target=writer, args=(w_id,))
            threads.append(t)

        # Launch 4 reader threads
        for r_id in range(4):
            t = threading.Thread(target=reader, args=(r_id,))
            threads.append(t)

        for t in threads:
            t.start()

        # Wait for writers to complete
        for t in threads[:4]:
            t.join(timeout=10.0)

        stop_event.set()
        for t in threads[4:]:
            t.join(timeout=5.0)

        rebalancer.stop()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
        assert len(index) > 0
        assert index.partition_count >= 1


class TestRegressionBugFixes:
    """Regression tests covering specific bugs found and fixed in the implementation."""

    def test_insert_returns_correct_pid_after_auto_split(self) -> None:
        """Bug: insert() was returning the stale parent partition ID after an auto-split.

        After a split the parent no longer exists in _partitions. The returned
        partition ID must be the child partition that actually holds the vector.
        """
        dim = 4
        # max=4 so the 5th insert triggers a split
        index = SPFreshHotIndex(
            dimension=dim, max_partition_size=4, min_partition_size=1, auto_rebalance=True
        )
        vecs = _random_normalized_vectors(6, dim)

        # First 4 inserts fill the partition (no split yet)
        for i in range(4):
            index.insert(f"v-{i}", vecs[i])

        # 5th insert triggers the auto-split
        pid_after_split = index.insert("v-4", vecs[4])

        # The returned pid must exist in the live partitions (bug was it returned stale parent)
        with index._lock:
            assert pid_after_split in index._partitions, (
                f"insert() returned stale partition ID '{pid_after_split}' "
                "which no longer exists after auto-split"
            )

        # All vectors must still be retrievable after split
        for i in range(5):
            assert index.get(f"v-{i}") is not None, f"v-{i} lost after auto-split"

    def test_merge_deferred_when_target_would_overflow(self) -> None:
        """Bug: merge_partition() had no overflow guard; it could push a partition
        above max_partition_size, which would cascade into infinite split-merge loops.

        When a merge would exceed max_partition_size it must be deferred (return None).
        """
        dim = 4
        max_sz = 4
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=max_sz,
            min_partition_size=1,
            auto_rebalance=False,
        )

        # Manually place 4 vectors in partition A
        vecs_a = _random_normalized_vectors(4, dim)

        for i, v in enumerate(vecs_a):
            index.insert(f"a-{i}", v)

        # Force a split so we have 2 partitions
        with index._lock:
            pid = list(index._partitions.keys())[0]
            index.split_partition(pid)

        assert index.partition_count == 2, "Expected 2 partitions after manual split"

        # Fill one of the child partitions to exactly max_sz
        with index._lock:
            child_pids = list(index._partitions.keys())
            # pick the smaller child, top it up to max_sz
            smaller_pid = min(child_pids, key=lambda p: index._partitions[p].size())
            target_pid = [p for p in child_pids if p != smaller_pid][0]

            # add vectors to smaller until both together exceed max_sz
            existing_target_size = index._partitions[target_pid].size()
            existing_smaller_size = index._partitions[smaller_pid].size()

        # The merge should be deferred if combined > max_sz
        if existing_target_size + existing_smaller_size > max_sz:
            with index._lock:
                result = index.merge_partition(smaller_pid)
            assert result is None, (
                "Merge should be deferred when combined size > max_partition_size"
            )

    def test_boundary_migration_skips_already_migrated_vectors(self) -> None:
        """Bug: boundary migration iterated a stale snapshot and could call
        part.remove(vid) on a vector that was already migrated in a prior
        iteration of the same loop, causing silent state corruption.

        After rebalance(), total vector count must equal what was inserted minus deletes.
        """
        dim = 4
        index = SPFreshHotIndex(
            dimension=dim,
            max_partition_size=20,
            min_partition_size=2,
            drift_threshold=0.01,  # very low threshold to trigger migration
            auto_rebalance=False,
        )

        # Populate and trigger a manual split so we have multiple partitions
        vecs = _random_normalized_vectors(10, dim)
        for i, v in enumerate(vecs):
            index.insert(f"item-{i}", v)

        with index._lock:
            pid = list(index._partitions.keys())[0]
            index.split_partition(pid)

        before_count = len(index)
        index.rebalance()
        after_count = len(index)

        # Total vector count must be conserved — no duplicates, no orphans
        assert before_count == after_count, (
            f"Vector count changed during rebalance: {before_count} -> {after_count}"
        )
        # All vectors must still be retrievable
        for i in range(10):
            assert index.get(f"item-{i}") is not None, f"item-{i} lost after rebalance"

    def test_config_validation_order_min_before_bounds(self) -> None:
        """Bug: old code checked max>min before checking min>=1.
        When min=0 and max=5 it would raise 'must be greater than' instead of
        the more informative 'min_partition_size must be >= 1' message.
        """
        with pytest.raises(ValueError, match="min_partition_size must be >= 1"):
            SPFreshConfig(min_partition_size=0, max_partition_size=5)

    def test_upsert_preserves_only_one_copy(self) -> None:
        """Inserting the same ID twice (upsert) must not create duplicate entries."""
        dim = 4
        index = SPFreshHotIndex(dimension=dim, max_partition_size=10, min_partition_size=2)

        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        index.insert("doc-x", v1, metadata={"ver": 1})
        index.insert("doc-x", v2, metadata={"ver": 2})

        assert len(index) == 1, "Upsert must not create duplicate entries"
        retrieved = index.get("doc-x")
        assert retrieved is not None
        _, meta, _ = retrieved
        assert meta.get("ver") == 2, "Upsert must update metadata to latest version"
