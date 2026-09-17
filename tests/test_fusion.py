import pytest

from dynavec.eval import LabeledQuery
from dynavec.fusion import RRFWeightFitter
from dynavec.models import SearchResult


def _make_result(doc_id: str, score: float) -> SearchResult:
    return SearchResult(id=doc_id, score=score)


def test_score_returns_float():
    # Two queries, two retrievers
    labeled = [
        LabeledQuery(query="q1", relevant_ids={"doc_a"}),
        LabeledQuery(query="q2", relevant_ids={"doc_b"}),
    ]

    # retriever 0 results per query: [q1_results, q2_results]
    dense = [
        [_make_result("doc_a", 0.9), _make_result("doc_b", 0.5)],  # query 0
        [_make_result("doc_b", 0.8), _make_result("doc_a", 0.3)],  # query 1
    ]
    # retriever 1 results per query
    sparse = [
        [_make_result("doc_a", 0.7), _make_result("doc_c", 0.2)],
        [_make_result("doc_b", 0.9), _make_result("doc_c", 0.1)],
    ]

    fitter = RRFWeightFitter(labeled, [dense, sparse])
    s = fitter.score([1.0, 1.0])
    assert 0.0 <= s <= 1.0

def test_fit_grid_improves_over_equal_weights():
    labeled = [
        LabeledQuery(query="q1", relevant_ids={"doc_a"}),
    ]
    # Dense retriever always puts doc_a first — it's the better retriever
    dense = [[_make_result("doc_a", 0.9), _make_result("doc_b", 0.1)]]
    # Sparse retriever buries doc_a — it's noisy
    sparse = [[_make_result("doc_b", 0.9), _make_result("doc_a", 0.1)]]

    fitter = RRFWeightFitter(labeled, [dense, sparse])
    best_weights = fitter.fit(method="grid", n_points=100)

    # Fitted weights must outperform pure sparse (which always buries doc_a)
    assert fitter.score(best_weights) > fitter.score([0.0, 1.0])

def test_fit_random_three_retrievers():
    labeled = [LabeledQuery(query="q1", relevant_ids={"doc_a"})]
    r1 = [[_make_result("doc_a", 0.9), _make_result("doc_b", 0.1)]]
    r2 = [[_make_result("doc_b", 0.9), _make_result("doc_a", 0.1)]]
    r3 = [[_make_result("doc_a", 0.7), _make_result("doc_b", 0.3)]]

    fitter = RRFWeightFitter(labeled, [r1, r2, r3])
    weights = fitter.fit(method="random", n_points=50)

    assert len(weights) == 3
    assert all(w >= 0 for w in weights)

def test_empty_labeled_queries_raises():
    with pytest.raises(ValueError, match="labeled_queries"):
        RRFWeightFitter([], [[]])

def test_unknown_method_raises():
    labeled = [LabeledQuery(query="q", relevant_ids={"d"})]
    fitter = RRFWeightFitter(labeled, [[[_make_result("d", 1.0)]]])
    with pytest.raises(ValueError, match="Unknown method"):
        fitter.fit(method="banana")
