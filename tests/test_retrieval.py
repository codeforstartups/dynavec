"""Tests for the retrieval algorithms (pure, no AWS)."""

from dynavec.models import SearchResult
from dynavec.retrieval import (
    distance_to_score,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)


def test_distance_to_score_cosine_monotonic():
    # smaller distance -> higher score
    assert distance_to_score(0.0, "cosine") == 1.0
    assert distance_to_score(0.2, "cosine") > distance_to_score(0.5, "cosine")


def test_distance_to_score_euclidean_bounded():
    assert distance_to_score(0.0, "euclidean") == 1.0
    assert 0.0 < distance_to_score(9.0, "euclidean") < distance_to_score(1.0, "euclidean")


def _r(i, distance=None):
    return SearchResult(id=i, score=0.0, distance=distance)


def test_rrf_prefers_consistently_top_ranked():
    dense = [_r("a"), _r("b"), _r("c")]
    sparse = [_r("b"), _r("a"), _r("d")]
    fused = reciprocal_rank_fusion([dense, sparse])
    # 'a' (ranks 1,2) and 'b' (ranks 2,1) should top the list
    top_two = {fused[0].id, fused[1].id}
    assert top_two == {"a", "b"}
    # scores are descending
    assert all(fused[i].score >= fused[i + 1].score for i in range(len(fused) - 1))


def test_rrf_weights_shift_ranking():
    dense = [_r("a"), _r("b")]
    sparse = [_r("b"), _r("a")]
    fused = reciprocal_rank_fusion([dense, sparse], weights=[3.0, 1.0])
    assert fused[0].id == "a"


def test_mmr_selects_diverse_items():
    q = [1.0, 0.0]
    # two near-duplicates pointing right, one pointing up
    cands = [
        SearchResult(id="dup1", score=0.0, vector=[1.0, 0.0]),
        SearchResult(id="dup2", score=0.0, vector=[0.99, 0.01]),
        SearchResult(id="orthogonal", score=0.0, vector=[0.0, 1.0]),
    ]
    # lambda_mult < 0.5 leans on diversity, so the 2nd pick should be the
    # orthogonal vector rather than the near-duplicate of the 1st pick.
    picked = maximal_marginal_relevance(q, cands, top_k=2, lambda_mult=0.3)
    ids = [p.id for p in picked]
    # first pick is the most relevant duplicate; second favors diversity
    assert ids[0] in ("dup1", "dup2")
    assert "orthogonal" in ids


def test_mmr_without_vectors_falls_back():
    cands = [SearchResult(id="a", score=0.0), SearchResult(id="b", score=0.0)]
    picked = maximal_marginal_relevance([1.0, 0.0], cands, top_k=1)
    assert len(picked) == 1
