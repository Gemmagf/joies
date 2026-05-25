"""Tests for the offline demo mode (rule-based intent + templated composer)."""

from __future__ import annotations

from maison_concierge.agents.demo_composer import compose_reply_templated
from maison_concierge.agents.demo_intent import classify_intent_rule_based
from maison_concierge.models import ClientIntent
from maison_concierge.retrieval import CatalogSearchResult, HeritageSearchResult
from maison_concierge.data_loader import load_catalog, load_heritage


def test_intent_appointment_en():
    r = classify_intent_rule_based("I'd like to book an appointment in Geneva")
    assert r.intent == "appointment"
    assert r.confidence >= 0.8
    assert r.locale == "en"
    assert not r.escalate


def test_intent_celebration_fr():
    r = classify_intent_rule_based("Cadeau d'anniversaire de mariage autour de 5000 CHF")
    assert r.intent == "celebration"
    assert r.locale == "fr"
    assert r.needs_catalog


def test_intent_heritage_fr():
    r = classify_intent_rule_based("D'où vient le motif Alhambra ?")
    assert r.intent == "heritage_inquiry"
    assert r.locale == "fr"
    assert r.needs_heritage


def test_intent_escalates_on_third_party_brand():
    r = classify_intent_rule_based("Do you carry Cartier or Bvlgari?")
    assert r.escalate
    assert r.intent == "unknown"


def test_intent_escalates_on_valuation_request():
    r = classify_intent_rule_based("What's the resale value of my grandmother's necklace?")
    assert r.escalate


def test_intent_greeting_does_not_escalate():
    r = classify_intent_rule_based("Hello.")
    assert not r.escalate
    assert r.intent == "unknown"


def test_composer_celebration_returns_named_pieces():
    pieces = load_catalog()[:2]
    hits = [CatalogSearchResult(piece=p, score=0.9 - i * 0.05) for i, p in enumerate(pieces)]
    reply, _ = compose_reply_templated(
        intent=ClientIntent.CELEBRATION,
        locale="en",
        catalog_hits=hits,
        heritage_hits=[],
        visual_hits=[],
    )
    assert pieces[0].id in reply
    assert pieces[1].id in reply
    assert "CHF" in reply


def test_composer_heritage_includes_citation():
    docs = load_heritage()[:1]
    hits = [HeritageSearchResult(document=d, score=0.9, snippet=d.body.en[:200]) for d in docs]
    reply, citations = compose_reply_templated(
        intent=ClientIntent.HERITAGE_INQUIRY,
        locale="en",
        catalog_hits=[],
        heritage_hits=hits,
        visual_hits=[],
    )
    assert f"[{docs[0].id}]" in reply
    assert len(citations) == 1
    assert citations[0].document_id == docs[0].id


def test_composer_price_inquiry_lists_options_fr():
    pieces = load_catalog()[:3]
    hits = [CatalogSearchResult(piece=p, score=0.9) for p in pieces]
    reply, _ = compose_reply_templated(
        intent=ClientIntent.PRICE_INQUIRY,
        locale="fr",
        catalog_hits=hits,
        heritage_hits=[],
        visual_hits=[],
    )
    assert all(p.id in reply for p in pieces)
    assert "CHF" in reply
