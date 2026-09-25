import json

import pytest

from dynavec.eval import LabeledQuery
from dynavec.fusion import FitResult, RRFWeightFitter
from dynavec.models import SearchResult


def _make_result(doc_id: str, score: float) -> SearchResult:
    return SearchResult(id=doc_id, score=score)


# ── helpers ─────────────────────────────────────────────────────────────────

def _two_retriever_fixture():
    """Two queries, two retrievers — dense is better."""
    labeled = [
        LabeledQuery(query="q1", relevant_ids={"doc_a"}),
        LabeledQuery(query="q2", relevant_ids={"doc_b"}),
    ]
    dense = [
        [_make_result("doc_a", 0.9), _make_result("doc_b", 0.5)],
        [_make_result("doc_b", 0.8), _make_result("doc_a", 0.3)],
    ]
    sparse = [
        [_make_result("doc_a", 0.7), _make_result("doc_c", 0.2)],
        [_make_result("doc_b", 0.9), _make_result("doc_c", 0.1)],
    ]
    return labeled, dense, sparse


def _three_retriever_fixture():
    """One query, three retrievers."""
    labeled = [LabeledQuery(query="q1", relevant_ids={"doc_a"})]
    r1 = [[_make_result("doc_a", 0.9), _make_result("doc_b", 0.1)]]
    r2 = [[_make_result("doc_b", 0.9), _make_result("doc_a", 0.1)]]
    r3 = [[_make_result("doc_a", 0.7), _make_result("doc_b", 0.3)]]
    return labeled, r1, r2, r3


# ── score ───────────────────────────────────────────────────────────────────

def test_score_returns_float():
    labeled, dense, sparse = _two_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [dense, sparse])
    s = fitter.score([1.0, 1.0])
    assert 0.0 <= s <= 1.0


# ── fit() returns FitResult ─────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["grid", "random"])
def test_fit_returns_fit_result(method):
    labeled, dense, sparse = _two_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [dense, sparse])
    result = fitter.fit(method=method, n_points=20)

    assert isinstance(result, FitResult)
    assert len(result.weights) == 2
    assert 0.0 <= result.score <= 1.0
    assert result.method == method
    assert result.n_evaluations > 0


# ── grid search ─────────────────────────────────────────────────────────────

def test_fit_grid_improves_over_equal_weights():
    labeled = [LabeledQuery(query="q1", relevant_ids={"doc_a"})]
    dense = [[_make_result("doc_a", 0.9), _make_result("doc_b", 0.1)]]
    sparse = [[_make_result("doc_b", 0.9), _make_result("doc_a", 0.1)]]

    fitter = RRFWeightFitter(labeled, [dense, sparse])
    result = fitter.fit(method="grid", n_points=100)

    assert result.score > fitter.score([0.0, 1.0])


def test_fit_grid_three_retrievers():
    """Grid search must work for 3+ retrievers (was broken before)."""
    labeled, r1, r2, r3 = _three_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [r1, r2, r3])
    result = fitter.fit(method="grid", n_points=10)

    assert isinstance(result, FitResult)
    assert len(result.weights) == 3
    assert all(w >= 0 for w in result.weights)
    assert abs(sum(result.weights) - 1.0) < 1e-9
    assert result.n_evaluations > 0


# ── random search ───────────────────────────────────────────────────────────

def test_fit_random_three_retrievers():
    labeled, r1, r2, r3 = _three_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [r1, r2, r3])
    result = fitter.fit(method="random", n_points=50)

    assert len(result.weights) == 3
    assert all(w >= 0 for w in result.weights)


# ── bayesian fitting ────────────────────────────────────────────────────────

def test_fit_bayesian_two_retrievers():
    """Bayesian fitting should return valid FitResult."""
    labeled, dense, sparse = _two_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [dense, sparse])
    result = fitter.fit(method="bayesian", n_points=50)

    assert isinstance(result, FitResult)
    assert result.method == "bayesian"
    assert len(result.weights) == 2
    assert 0.0 <= result.score <= 1.0
    assert all(w >= 0 for w in result.weights)
    assert abs(sum(result.weights) - 1.0) < 1e-6


def test_fit_bayesian_three_retrievers():
    labeled, r1, r2, r3 = _three_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [r1, r2, r3])
    result = fitter.fit(method="bayesian", n_points=50)

    assert isinstance(result, FitResult)
    assert len(result.weights) == 3
    assert abs(sum(result.weights) - 1.0) < 1e-6


# ── serialisation (save / load) ─────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    """FitResult should survive a JSON roundtrip."""
    original = FitResult(
        weights=[0.7, 0.3],
        score=0.85,
        method="grid",
        n_evaluations=101,
    )
    path = tmp_path / "fit_result.json"
    original.save(path)

    loaded = FitResult.load(path)
    assert loaded.weights == original.weights
    assert loaded.score == pytest.approx(original.score)
    assert loaded.method == original.method
    assert loaded.n_evaluations == original.n_evaluations


def test_save_creates_valid_json(tmp_path):
    result = FitResult(weights=[0.6, 0.4], score=0.9, method="random", n_evaluations=50)
    path = tmp_path / "result.json"
    result.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["weights"] == [0.6, 0.4]
    assert data["score"] == 0.9
    assert data["method"] == "random"
    assert data["n_evaluations"] == 50


def test_to_dict():
    result = FitResult(weights=[0.5, 0.5], score=0.8, method="grid", n_evaluations=10)
    d = result.to_dict()
    assert d == {"weights": [0.5, 0.5], "score": 0.8, "method": "grid", "n_evaluations": 10}


# ── end-to-end: fit then save/load ──────────────────────────────────────────

def test_fit_and_save_load(tmp_path):
    """Full workflow: fit → save → load → use weights."""
    labeled, dense, sparse = _two_retriever_fixture()
    fitter = RRFWeightFitter(labeled, [dense, sparse])

    result = fitter.fit(method="grid", n_points=50)
    path = tmp_path / "weights.json"
    result.save(path)

    loaded = FitResult.load(path)
    # The loaded weights should produce the same score
    assert fitter.score(loaded.weights) == pytest.approx(result.score)


# ── error handling ──────────────────────────────────────────────────────────

def test_empty_labeled_queries_raises():
    with pytest.raises(ValueError, match="labeled_queries"):
        RRFWeightFitter([], [[]])


def test_unknown_method_raises():
    labeled = [LabeledQuery(query="q", relevant_ids={"d"})]
    fitter = RRFWeightFitter(labeled, [[[_make_result("d", 1.0)]]])
    with pytest.raises(ValueError, match="Unknown method"):
        fitter.fit(method="banana")
