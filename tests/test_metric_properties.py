"""Property-based tests for metrics (vector similarity/distance and IR evaluation).

Fuzzes metric monotonicity, ordering, symmetry, scale invariance, and ranking
invariants using Hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dynavec.eval.retrieval import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from dynavec.metrics import (
    composite_score,
    normalize_scores,
    rescore,
    score,
)

# ---------------------------------------------------------------------------
# Hypothesis Strategies: Vector Metrics
# ---------------------------------------------------------------------------

_VALID_METRICS = ("cosine", "dot", "euclidean", "manhattan")


def _floats():
    """Finite, bounded float strategy to avoid extreme overflow/underflow."""
    return st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    )


@st.composite
def _vector_and_matrix(
    draw,
    min_dim: int = 2,
    max_dim: int = 16,
    min_rows: int = 1,
    max_rows: int = 12,
):
    """Draw a query vector and a candidate matrix of matching dimension."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    q = draw(
        arrays(
            dtype=np.float32,
            shape=(dim,),
            elements=_floats(),
        )
    )
    mat = draw(
        arrays(
            dtype=np.float32,
            shape=(rows, dim),
            elements=_floats(),
        )
    )
    return q, mat


@st.composite
def _non_zero_vector(draw, min_dim: int = 2, max_dim: int = 16):
    """Draw a non-zero vector with sufficient L2 norm."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    vec = draw(
        arrays(
            dtype=np.float32,
            shape=(dim,),
            elements=_floats(),
        )
    )
    assume(float(np.linalg.norm(vec)) > 1e-4)
    return vec


@st.composite
def _two_non_zero_vectors(draw, min_dim: int = 2, max_dim: int = 8):
    """Draw two non-zero vectors guaranteed to have the exact same dimension."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    v1 = draw(
        arrays(
            dtype=np.float32,
            shape=(dim,),
            elements=_floats(),
        )
    )
    v2 = draw(
        arrays(
            dtype=np.float32,
            shape=(dim,),
            elements=_floats(),
        )
    )
    assume(float(np.linalg.norm(v1)) > 1e-4)
    assume(float(np.linalg.norm(v2)) > 1e-4)
    return v1, v2


@st.composite
def _composite_weights(draw):
    """Draw valid weight dict with positive weights for a non-empty subset of metrics."""
    selected_metrics = draw(
        st.sets(
            st.sampled_from(_VALID_METRICS),
            min_size=1,
            max_size=4,
        )
    )
    weights = {}
    for m in selected_metrics:
        weights[m] = draw(
            st.floats(
                min_value=0.01,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    return weights


# ---------------------------------------------------------------------------
# Vector Metrics Properties (dynavec.metrics)
# ---------------------------------------------------------------------------


@given(
    data=_vector_and_matrix(min_rows=1, max_rows=10),
    metric=st.sampled_from(_VALID_METRICS),
)
@settings(max_examples=150)
def test_score_output_shape_and_finite(data, metric):
    """score() always returns a 1D float array matching the candidate matrix row count."""
    q, mat = data
    scores = score(q, mat, metric)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (mat.shape[0],)
    assert np.all(np.isfinite(scores))


@given(
    v=_non_zero_vector(),
    metric=st.sampled_from(("euclidean", "manhattan", "cosine")),
)
@settings(max_examples=100)
def test_score_self_similarity_is_maximal(v, metric):
    """Comparing any non-zero vector to itself gives the maximal similarity score (1.0)."""
    s_self = float(score(v, v, metric)[0])
    assert s_self == pytest.approx(1.0, abs=1e-4)


@given(
    vec_pair=_two_non_zero_vectors(),
    metric=st.sampled_from(_VALID_METRICS),
)
@settings(max_examples=100)
def test_score_symmetry(vec_pair, metric):
    """score(u, v) == score(v, u) for all supported similarity and distance metrics."""
    u, v = vec_pair
    s_uv = float(score(u, v, metric)[0])
    s_vu = float(score(v, u, metric)[0])
    assert s_uv == pytest.approx(s_vu, rel=1e-4, abs=1e-4)


@given(
    vec_pair=_two_non_zero_vectors(min_dim=2, max_dim=8),
    scale_a=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_euclidean_and_manhattan_distance_monotonicity(vec_pair, scale_a, delta):
    """As a candidate vector moves strictly further along a displacement ray,

    Euclidean and Manhattan similarity scores strictly decrease monotonically.
    """
    q, displacement = vec_pair
    scale_b = scale_a + delta  # strictly further away

    cand_near = q + scale_a * displacement
    cand_far = q + scale_b * displacement
    mat = np.stack([cand_near, cand_far], axis=0)

    for metric in ("euclidean", "manhattan"):
        scores = score(q, mat, metric)
        # Closer candidate (near) must have strictly higher similarity score than farther candidate
        assert scores[0] > scores[1], f"Metric {metric} failed monotonicity: {scores[0]} <= {scores[1]}"


@given(
    q=_non_zero_vector(min_dim=2, max_dim=8),
    alpha_1=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_dot_product_magnitude_monotonicity(q, alpha_1, delta):
    """For vectors aligned in the direction of query q,

    larger magnitude strictly yields higher dot product score.
    """
    alpha_2 = alpha_1 + delta  # larger magnitude
    v1 = alpha_1 * q
    v2 = alpha_2 * q
    mat = np.stack([v1, v2], axis=0)

    scores = score(q, mat, "dot")
    assert scores[1] > scores[0]


@given(
    vec_pair=_two_non_zero_vectors(min_dim=2, max_dim=8),
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_cosine_scale_invariance(vec_pair, scale):
    """Cosine similarity is scale-invariant: scaling a vector by any positive alpha

    leaves its cosine similarity to the query unchanged.
    """
    q, cand = vec_pair
    cand_scaled = cand * scale

    mat = np.stack([cand, cand_scaled], axis=0)
    scores = score(q, mat, "cosine")

    assert scores[0] == pytest.approx(scores[1], abs=1e-4)


@given(
    data=_vector_and_matrix(min_rows=1, max_rows=10),
)
@settings(max_examples=100)
def test_metric_score_ranges(data):
    """Euclidean & Manhattan scores are strictly in (0, 1], and Cosine is in [-1, 1]."""
    q, mat = data

    for metric in ("euclidean", "manhattan"):
        scores = score(q, mat, metric)
        assert np.all(scores > 0.0)
        assert np.all(scores <= 1.0 + 1e-6)

    cos_scores = score(q, mat, "cosine")
    assert np.all(cos_scores >= -1.0 - 1e-4)
    assert np.all(cos_scores <= 1.0 + 1e-4)


# ---------------------------------------------------------------------------
# Normalization Properties (dynavec.metrics.normalize_scores)
# ---------------------------------------------------------------------------


@given(
    scores=arrays(
        dtype=np.float32,
        shape=st.integers(min_value=1, max_value=30),
        elements=_floats(),
    )
)
@settings(max_examples=150)
def test_normalize_scores_bounds_and_order_preservation(scores):
    """normalize_scores always maps to [0, 1] and strictly preserves relative order."""
    normalized = normalize_scores(scores)

    assert normalized.shape == scores.shape
    assert np.all(normalized >= -1e-6)
    assert np.all(normalized <= 1.0 + 1e-6)

    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        assert np.all(normalized == 0.0)
    else:
        assert float(normalized.min()) == pytest.approx(0.0, abs=1e-6)
        assert float(normalized.max()) == pytest.approx(1.0, abs=1e-6)

        # Monotonicity / Order preservation:
        # If scores[i] < scores[j], then normalized[i] <= normalized[j]
        # and ranking under argsort matches (up to ties)
        order_orig = np.argsort(scores)
        sorted_norm = normalized[order_orig]
        assert np.all(np.diff(sorted_norm) >= -1e-7), "Order preservation violated"


@given(
    scores=arrays(
        dtype=np.float32,
        shape=st.integers(min_value=2, max_value=20),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_normalize_scores_affine_invariance_and_idempotence(scores, scale, shift):
    """normalize_scores is affine-invariant (scale > 0) and idempotent."""
    assume(float(scores.max()) - float(scores.min()) > 1e-5)

    transformed = scores * scale + shift
    norm_orig = normalize_scores(scores)
    norm_transformed = normalize_scores(transformed)

    assert norm_orig == pytest.approx(norm_transformed, abs=1e-4)

    # Idempotence: normalize_scores(normalize_scores(x)) == normalize_scores(x)
    norm_idempotent = normalize_scores(norm_orig)
    assert norm_idempotent == pytest.approx(norm_orig, abs=1e-4)


# ---------------------------------------------------------------------------
# Composite Scoring & Rescoring Properties (dynavec.metrics)
# ---------------------------------------------------------------------------


@given(
    data=_vector_and_matrix(min_rows=2, max_rows=10),
    weights=_composite_weights(),
)
@settings(max_examples=100)
def test_composite_score_bounds_and_weight_scale_invariance(data, weights):
    """composite_score is bounded in [0, 1] and invariant to scaling all weights by c > 0."""
    q, mat = data
    comp = composite_score(q, mat, weights)

    assert comp.shape == (mat.shape[0],)
    assert np.all(comp >= -1e-6)
    assert np.all(comp <= 1.0 + 1e-6)

    # Scale weights by constant c > 0
    scaled_weights = {k: v * 3.7 for k, v in weights.items()}
    comp_scaled = composite_score(q, mat, scaled_weights)
    assert comp == pytest.approx(comp_scaled, abs=1e-5)


@given(
    data=_vector_and_matrix(min_rows=2, max_rows=10),
    metric=st.sampled_from(_VALID_METRICS),
    weight=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_composite_score_single_metric_equivalence(data, metric, weight):
    """composite_score with a single metric equals normalize_scores(score(q, mat, metric))."""
    q, mat = data
    comp = composite_score(q, mat, {metric: weight})
    expected = normalize_scores(score(q, mat, metric))
    assert comp == pytest.approx(expected, abs=1e-5)


@given(
    data=_vector_and_matrix(min_rows=2, max_rows=12),
    metric=st.sampled_from(_VALID_METRICS),
    normalize=st.booleans(),
)
@settings(max_examples=100)
def test_rescore_permutation_and_ordering_invariants(data, metric, normalize):
    """rescore returns a valid permutation of candidate indices and monotonically

    non-increasing scores at the ordered indices.
    """
    q, mat = data
    order, scores = rescore(q, mat, metric, normalize=normalize)
    n = mat.shape[0]

    # 1. Order is a valid permutation of 0..n-1
    assert sorted(list(order)) == list(range(n))

    # 2. Scores at ordered indices must be monotonically non-increasing
    ordered_scores = scores[order]
    assert np.all(np.diff(ordered_scores) <= 1e-6), f"Scores not non-increasing: {ordered_scores}"

    # 3. Top element has maximal score
    assert scores[order[0]] == pytest.approx(float(scores.max()), abs=1e-5)

    # 4. If normalized, scores are in [0, 1]
    if normalize:
        assert np.all(scores >= -1e-6)
        assert np.all(scores <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Hypothesis Strategies: Information Retrieval (IR) Metrics
# ---------------------------------------------------------------------------

_DOC_ALPHABET = st.sampled_from([f"doc_{i}" for i in range(1, 30)])


@st.composite
def _retrieval_scenario(draw, min_retrieved: int = 1, max_retrieved: int = 20):
    """Draw retrieved document IDs and relevant ground truth IDs."""
    retrieved = draw(
        st.lists(
            _DOC_ALPHABET,
            min_size=min_retrieved,
            max_size=max_retrieved,
            unique=True,
        )
    )
    relevant = draw(
        st.sets(
            _DOC_ALPHABET,
            min_size=1,
            max_size=15,
        )
    )
    k = draw(st.integers(min_value=1, max_value=len(retrieved) + 5))
    return retrieved, relevant, k


# ---------------------------------------------------------------------------
# IR Metrics Properties (dynavec.eval.retrieval)
# ---------------------------------------------------------------------------


@given(scenario=_retrieval_scenario())
@settings(max_examples=150)
def test_ir_metrics_boundedness_in_unit_interval(scenario):
    """All IR metrics (Recall@k, Precision@k, MRR, nDCG@k) are bounded in [0.0, 1.0]."""
    retrieved, relevant, k = scenario

    rec = compute_recall_at_k(retrieved, relevant, k)
    prec = compute_precision_at_k(retrieved, relevant, k)
    mrr = compute_mrr(retrieved, relevant, k)
    ndcg = compute_ndcg_at_k(retrieved, relevant, k)

    for name, val in [("Recall", rec), ("Precision", prec), ("MRR", mrr), ("nDCG", ndcg)]:
        assert 0.0 <= val <= 1.0, f"{name}@k={k} out of bounds: {val}"


@given(
    retrieved=st.lists(_DOC_ALPHABET, min_size=3, max_size=20, unique=True),
    relevant=st.sets(_DOC_ALPHABET, min_size=1, max_size=10),
    k1=st.integers(min_value=1, max_value=10),
    k2=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=150)
def test_recall_at_k_monotonicity_with_respect_to_k(retrieved, relevant, k1, k2):
    """Recall@k is monotonically non-decreasing with respect to cutoff k: k1 <= k2 => Recall@k1 <= Recall@k2."""
    assume(k1 <= k2)

    rec_1 = compute_recall_at_k(retrieved, relevant, k1)
    rec_2 = compute_recall_at_k(retrieved, relevant, k2)

    assert rec_1 <= rec_2 + 1e-6, f"Recall not monotonic: Recall@{k1}={rec_1} > Recall@{k2}={rec_2}"


@given(
    retrieved=st.lists(_DOC_ALPHABET, min_size=1, max_size=20, unique=True),
    relevant=st.sets(_DOC_ALPHABET, min_size=1, max_size=10),
    k=st.integers(min_value=1, max_value=25),
)
@settings(max_examples=100)
def test_precision_at_k_upper_bound(retrieved, relevant, k):
    """Precision@k cannot exceed min(1.0, |relevant| / k)."""
    prec = compute_precision_at_k(retrieved, relevant, k)
    max_possible = min(1.0, len(relevant) / k)
    assert prec <= max_possible + 1e-4


@given(
    docs=st.lists(_DOC_ALPHABET, min_size=3, max_size=15, unique=True),
    target_pos_early=st.integers(min_value=0, max_value=5),
    gap=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_mrr_rank_monotonicity(docs, target_pos_early, gap):
    """Moving the first relevant document to an earlier rank strictly increases MRR."""
    target_pos_late = target_pos_early + gap
    assume(target_pos_late < len(docs))

    target_doc = docs[target_pos_late]
    relevant = {target_doc}

    # List where target doc is at target_pos_late
    list_late = list(docs)

    # List where target doc is swapped to target_pos_early
    list_early = list(docs)
    list_early[target_pos_early], list_early[target_pos_late] = (
        list_early[target_pos_late],
        list_early[target_pos_early],
    )

    mrr_early = compute_mrr(list_early, relevant)
    mrr_late = compute_mrr(list_late, relevant)

    assert mrr_early > mrr_late, f"MRR early ({mrr_early}) should be > MRR late ({mrr_late})"


@given(
    relevant=st.sets(_DOC_ALPHABET, min_size=2, max_size=8),
    irrelevant=st.sets(_DOC_ALPHABET, min_size=2, max_size=8),
)
@settings(max_examples=100)
def test_ndcg_ideal_vs_disjoint_ranking(relevant, irrelevant):
    """Ideal ranking produces nDCG@k == 1.0, while disjoint ranking produces nDCG@k == 0.0."""
    assume(relevant.isdisjoint(irrelevant))

    # Perfect ranking: relevant items first
    ideal_retrieved = list(relevant) + list(irrelevant)
    k = len(relevant)

    ndcg_ideal = compute_ndcg_at_k(ideal_retrieved, relevant, k)
    assert ndcg_ideal == pytest.approx(1.0, abs=1e-4)

    # Disjoint ranking: only irrelevant items retrieved
    disjoint_retrieved = list(irrelevant)
    ndcg_disjoint = compute_ndcg_at_k(disjoint_retrieved, relevant, k)
    assert ndcg_disjoint == pytest.approx(0.0, abs=1e-4)


@given(
    retrieved=st.lists(_DOC_ALPHABET, min_size=1, max_size=10, unique=True),
)
@settings(max_examples=50)
def test_ir_metrics_empty_relevant_returns_zero(retrieved):
    """When relevant_ids is empty or k <= 0, all IR metrics return 0.0."""
    empty_rel = set()
    assert compute_recall_at_k(retrieved, empty_rel, 5) == 0.0
    assert compute_precision_at_k(retrieved, empty_rel, 5) == 0.0
    assert compute_mrr(retrieved, empty_rel, 5) == 0.0
    assert compute_ndcg_at_k(retrieved, empty_rel, 5) == 0.0

    # k <= 0
    some_rel = {retrieved[0]}
    assert compute_recall_at_k(retrieved, some_rel, 0) == 0.0
    assert compute_precision_at_k(retrieved, some_rel, 0) == 0.0
    assert compute_ndcg_at_k(retrieved, some_rel, 0) == 0.0
