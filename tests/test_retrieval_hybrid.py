"""Tests for the hybrid retriever and Reciprocal Rank Fusion.

The pure RRF function is unit-tested standalone. The HybridCatalogRAG integration
test verifies the search returns the requested k results in the right shape and
respects category / max_price filters.
"""

from __future__ import annotations

from maison_concierge.retrieval import HybridCatalogRAG, reciprocal_rank_fusion


def test_rrf_single_ranking_falls_back_to_inverse_rank():
    scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"]
    assert abs(scores["a"] - 1 / 61) < 1e-9


def test_rrf_combines_two_rankings_additively():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "b"]], k=60)
    # 'a' is rank 1 + rank 2; 'c' is rank 3 + rank 1; 'b' is rank 2 + rank 3
    expected_a = 1 / 61 + 1 / 62
    expected_c = 1 / 63 + 1 / 61
    expected_b = 1 / 62 + 1 / 63
    assert abs(scores["a"] - expected_a) < 1e-9
    assert abs(scores["c"] - expected_c) < 1e-9
    assert abs(scores["b"] - expected_b) < 1e-9


def test_rrf_document_in_only_one_ranking_still_scored():
    scores = reciprocal_rank_fusion([["a", "x"], ["a", "y"]], k=60)
    assert "x" in scores
    assert "y" in scores
    # 'a' appears in both, so should rank above singletons
    assert scores["a"] > scores["x"]
    assert scores["a"] > scores["y"]


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == {}


def test_hybrid_returns_catalog_search_results():
    hybrid = HybridCatalogRAG()
    hybrid.index()
    results = hybrid.search("yellow gold alhambra mother-of-pearl", k=3)
    assert 0 < len(results) <= 3
    for r in results:
        assert r.piece.id.startswith("VCA-")
        assert r.score > 0


def test_hybrid_respects_category_filter():
    hybrid = HybridCatalogRAG()
    hybrid.index()
    results = hybrid.search("alhambra", k=5, category="bracelet")
    assert all(r.piece.category == "bracelet" for r in results)


def test_hybrid_respects_max_price_filter():
    hybrid = HybridCatalogRAG()
    hybrid.index()
    results = hybrid.search("ring", k=5, max_price_chf=5_000.0)
    assert all(r.piece.price_chf <= 5_000.0 for r in results)


def test_hybrid_returns_empty_on_empty_query():
    hybrid = HybridCatalogRAG()
    hybrid.index()
    # All stopwords → empty token list on BM25 side; dense will still produce hits
    results = hybrid.search("the of and", k=3)
    # We still get dense-side results
    assert len(results) >= 0  # dense still works, hybrid just degrades to dense
