"""Tests for client-side metrics + composite rescoring (pure, no AWS)."""

import numpy as np
import pytest

from dynavec.metrics import composite_score, normalize_scores, rescore, score


@pytest.fixture
def data():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mat = np.array(
        [
            [1.0, 0.0, 0.0],   # identical
            [0.0, 1.0, 0.0],   # orthogonal
            [2.0, 0.0, 0.0],   # same direction, larger magnitude
        ],
        dtype=np.float32,
    )
    return q, mat


def test_cosine_ignores_magnitude(data):
    q, mat = data
    s = score(q, mat, "cosine")
    # identical and 2x-magnitude both point the same way -> cosine ~1 for both
    assert s[0] == pytest.approx(1.0, abs=1e-5)
    assert s[2] == pytest.approx(1.0, abs=1e-5)
    assert s[1] == pytest.approx(0.0, abs=1e-5)


def test_dot_rewards_magnitude(data):
    q, mat = data
    s = score(q, mat, "dot")
    # dot product favors the larger-magnitude aligned vector
    assert s[2] > s[0] > s[1]


def test_euclidean_and_manhattan_higher_is_closer(data):
    q, mat = data
    for metric in ("euclidean", "manhattan"):
        s = score(q, mat, metric)
        assert np.argmax(s) == 0  # identical vector is closest


def test_unknown_metric_raises(data):
    q, mat = data
    with pytest.raises(ValueError):
        score(q, mat, "cosmic")


def test_composite_normalizes_and_weights(data):
    q, mat = data
    s = composite_score(q, mat, {"cosine": 0.5, "manhattan": 0.5})
    assert len(s) == 3
    assert np.argmax(s) == 0


def test_rescore_returns_order_and_scores(data):
    q, mat = data
    order, scores = rescore(q, mat, "dot")
    assert list(order)[0] == 2  # largest magnitude first under dot
    assert len(scores) == 3


def test_normalize_scores_maps_result_set_to_unit_interval():
    normalized = normalize_scores(np.array([-2.0, 0.0, 6.0], dtype=np.float32))
    assert normalized == pytest.approx([0.0, 0.25, 1.0])
    assert normalize_scores(np.array([4.0, 4.0])) == pytest.approx([0.0, 0.0])
    assert normalize_scores(np.array([])).size == 0


def test_rescore_can_return_normalized_scores(data):
    q, mat = data
    order, scores = rescore(q, mat, "dot", normalize=True)
    assert list(order)[0] == 2
    assert scores == pytest.approx([0.5, 0.0, 1.0])
