# SPFresh-Style Incremental Hot Index Rebalancing

## 1. Overview & Problem Statement

dynavec's in-memory hot tier acts as a low-latency caching and query acceleration layer in front of durable cloud storage (S3 Vectors and DynamoDB).

In dynamic vector search workloads, datasets continuously ingest new vectors and delete obsolete ones. Conventional space-partitioning approaches (e.g., standard IVF, SPANN, or static k-means clustering) suffer from several limitations under dynamic updates:

1. **Partition Imbalance:** Hot clusters grow disproportionately large, creating high latency outliers during routing search.
2. **Centroid Drift:** Deletions and additions shift cluster centers over time, causing query routing to miss true nearest neighbors.
3. **Expensive Re-indexing Stops:** Traditional indexes require periodic full offline rebuilds, causing CPU spikes and latency degradation.

Following the principles of the **SPFresh** algorithm (Incremental In-Place Space Partitioning for Approximate Nearest Neighbor Search), dynavec implements an incremental, in-place dynamic partition manager that maintains index health with continuous splits, merges, and centroid drift tracking without full re-indexing.

---

## 2. Architecture & Data Structures

```
                           +------------------------+
                           |     SPFreshHotIndex    |
                           +------------------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
        +------------------+                       +-------------------+
        |  Routing Table   |                       | Background Worker |
        | (Centroids C_1..k|                       | (SPFreshRebalancer|
        +------------------+                       +-------------------+
                 |                                           |
       +---------+---------+                       +---------+---------+
       |                   |                       |                   |
+---------------+   +---------------+       +---------------+   +---------------+
| Partition P_1 |   | Partition P_2 |  ...  | Partition P_k |   | Partition P_m |
| - centroid    |   | - centroid    |       | - centroid    |   | - centroid    |
| - vectors     |   | - vectors     |       | - vectors     |   | - vectors     |
| - lock (RLock)|   | - lock (RLock)|       | - lock (RLock)|   | - lock (RLock)|
+---------------+   +---------------+       +---------------+   +---------------+
```

### Core Components

1. **`Partition`**:
   * **Centroid ($c_p \in \mathbb{R}^d$):** The representative center of the cluster.
   * **Anchor Centroid ($c_{anchor} \in \mathbb{R}^d$):** Baseline center snapshot to measure accumulated drift.
   * **Posting List ($V_p = \{v_1, v_2, \dots, v_n\}$):** Vector array and corresponding metadata mapping.
   * **Fine-Grained Concurrency:** Each partition maintains an independent lock (`threading.RLock`) to allow parallel read/write access across different clusters.

2. **`SPFreshHotIndex`**:
   * Manages the global collection of partitions.
   * Routes incoming queries and insertions to the nearest $n\_probe$ partitions based on cosine or Euclidean distance to partition centroids.
   * Monitors partition sizes and triggers dynamic rebalancing.

3. **`SPFreshRebalancer`**:
   * Orchestrates synchronous (on write) or asynchronous background maintenance cycles.
   * Executes partition splits, merges, and drift-based vector migration.

---

## 3. Dynamic Rebalancing Protocol

### 3.1 Dynamic Partition Splitting

When insertions push a partition's size beyond an upper capacity limit:
$$\text{Size}(P_k) > N_{\max}$$

The partition undergoes an in-place split:
1. **Bipartitioning (2-Means):** Two seed vectors with maximum distance within $P_k$ are selected as initial seeds $c_A$ and $c_B$.
2. **K-Means Iteration ($k=2$):** Fast 2-means clustering partitions the vectors into two sets $S_A$ and $S_B$.
3. **Partition Replacement:** The original partition $P_k$ is atomically replaced by two new child partitions $P_A$ and $P_B$ with updated centroids $c_A = \frac{1}{|S_A|}\sum_{v \in S_A} v$ and $c_B = \frac{1}{|S_B|}\sum_{v \in S_B} v$.
4. **Routing Table Update:** The global routing index updates its centroid list atomically.

### 3.2 Dynamic Partition Merging

When sustained deletions reduce a partition's size below a lower threshold:
$$\text{Size}(P_k) < N_{\min} \quad \text{and} \quad |\{\text{partitions}\}| > 1$$

The partition is consolidated:
1. **Nearest Centroid Discovery:** Find neighbor partition $P_j$ such that:
   $$P_j = \arg\min_{i \neq k} \mathcal{D}(c_k, c_i)$$
2. **Capacity Validation:** Ensure $\text{Size}(P_k) + \text{Size}(P_j) \le N_{\max}$. If this would exceed $N_{\max}$, merge is deferred until neighboring conditions change.
3. **Consolidation:** Move all vectors from $P_k$ into $P_j$, update the centroid of $P_j$, and atomically deregister $P_k$.

### 3.3 Centroid Drift Tracking & Migration

As vectors are inserted and deleted, the true geometric mean of vectors in $P_k$ shifts:
$$c_{\text{actual}} = \frac{1}{|P_k|} \sum_{v \in P_k} v$$

We track the normalized drift distance relative to the anchor centroid:
$$\Delta_{\text{drift}} = \frac{\|c_{\text{actual}} - c_{\text{anchor}}\|_2}{\|c_{\text{anchor}}\|_2 + \epsilon}$$

When $\Delta_{\text{drift}} > \theta_{\text{drift}}$ (configurable threshold, default 0.25):
* Centroid $c_k$ is updated to $c_{\text{actual}}$.
* $c_{\text{anchor}} \leftarrow c_{\text{actual}}$.
* Boundary vectors that now sit closer to a neighboring partition's centroid are migrated during the maintenance pass.

---

## 4. Query Routing & Search

1. **Query Centroid Distance:** Given query vector $q \in \mathbb{R}^d$, compute distance to all active partition centroids $\{c_1, c_2, \dots, c_m\}$.
2. **Probe Top-$k$ Partitions:** Select the $n\_probe$ nearest partitions:
   $$\mathcal{P}_{\text{probe}} = \text{Top-}(n\_probe) \left( \mathcal{D}(q, c_i) \right)$$
3. **Intra-Partition Exact Scan:** Scan vectors inside the selected partitions using vectorized NumPy dot products/distances.
4. **Result Aggregation:** Combine top-scoring vectors across probed partitions and return top-$K$ global results.

---

## 5. Thread Safety & Concurrency Model

* **Global Read-Lock Free Routing:** The list of active centroids uses copy-on-write or atomic reference swapping so read queries never block on centroid lookups.
* **Per-Partition Lock:** Reading/searching inside partition $P_i$ acquires a shared read lock on $P_i$. Inserting or deleting into $P_i$ acquires an exclusive write lock on $P_i$.
* **Non-Blocking Queries:** Searching in partition $P_1$ proceeds concurrently with an insertion or split occurring in partition $P_2$.

---

## 6. Integration with dynavec Hot/Cold Storage

```
[Write Ingestion]
       │
       ▼
[SPFresh Hot Tier Index] ──(Auto-split / Auto-merge)──> [Dynamic Memory Partitions]
       │
       ▼ (Flush / Sync)
[S3 Vectors / DynamoDB Storage]
```

* **Hot Ingest:** New vectors enter `SPFreshHotIndex` immediately for sub-millisecond query availability.
* **Background Persistence:** Partitions or delta logs are streamed asynchronously to S3 Vectors and DynamoDB for cold persistence and disaster recovery.

