"""Rule-based intent classifier."""

from __future__ import annotations

import pytest

from maison_concierge.hospitality.intent import (
    HospitalityIntent,
    classify_intent_rule_based,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("what time does the spa open?", HospitalityIntent.KNOWLEDGE),
        ("can I bring my dog?", HospitalityIntent.KNOWLEDGE),
        ("what is your cancellation policy?", HospitalityIntent.KNOWLEDGE),
        ("dress code at the restaurant?", HospitalityIntent.KNOWLEDGE),
        ("am I likely to cancel?", HospitalityIntent.PROFILE),
        ("what is my segment?", HospitalityIntent.PROFILE),
        ("who am I?", HospitalityIntent.PROFILE),
        ("change my booking to next week", HospitalityIntent.BOOKING),
        ("cancel my reservation", HospitalityIntent.BOOKING),
        ("what should I do tomorrow?", HospitalityIntent.RECOMMENDATION),
        ("recommend a restaurant", HospitalityIntent.RECOMMENDATION),
        ("this is unacceptable, I want a manager", HospitalityIntent.ESCALATE),
        ("I'm having a medical emergency", HospitalityIntent.ESCALATE),
        ("hello", HospitalityIntent.UNKNOWN),
    ],
)
def test_intent_routing(text: str, expected: HospitalityIntent) -> None:
    result = classify_intent_rule_based(text)
    assert result.intent == expected, f"{text!r} → {result.intent} (expected {expected})"


def test_escalation_flags_are_set() -> None:
    r = classify_intent_rule_based("this is terrible, get me the manager")
    assert r.escalate is True
    assert r.intent == HospitalityIntent.ESCALATE


def test_unknown_has_zero_confidence() -> None:
    r = classify_intent_rule_based("hello")
    assert r.intent == HospitalityIntent.UNKNOWN
    assert r.confidence == 0.0
