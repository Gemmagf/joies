"""Property KB — pages parse, embed, and return citation-ready hits."""

from __future__ import annotations

from maison_concierge.hospitality.kb import PropertyKB, load_kb_pages, page_by_id


def test_all_pages_parse_with_required_frontmatter() -> None:
    pages = load_kb_pages()
    assert len(pages) >= 10
    for p in pages:
        assert p.id.startswith("kb-")
        assert p.title
        assert p.body.strip()
        assert p.property_scope in {"lisboa", "algarve", "both"}


def test_page_by_id_returns_known_pages() -> None:
    assert page_by_id("kb-spa-wellness") is not None
    assert page_by_id("kb-cancellation") is not None
    assert page_by_id("kb-does-not-exist") is None


def test_kb_search_returns_relevant_hits_for_spa_query() -> None:
    kb = PropertyKB()
    hits = kb.search("what time does the spa open?", k=3)
    assert hits
    assert hits[0].page.id == "kb-spa-wellness"
    assert 0 <= hits[0].score <= 1


def test_kb_search_returns_cancellation_page_for_cancellation_query() -> None:
    kb = PropertyKB()
    hits = kb.search("cancellation policy for non-refundable deposit", k=3)
    assert hits
    top_ids = [h.page.id for h in hits]
    assert "kb-cancellation" in top_ids


def test_property_scope_filters_correctly() -> None:
    kb = PropertyKB()
    # A resort-only feature: the kids club at Algarve is described only in
    # kb-family-services (scope=both) and kb-property-algarve (scope=algarve).
    # Filtering to Lisboa should not surface the Algarve-specific page.
    hits_lisboa = kb.search("kids club", k=5, property_scope="lisboa")
    for h in hits_lisboa:
        assert h.page.property_scope in {"lisboa", "both"}
