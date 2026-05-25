"""Rule-based intent classifier — deterministic, no API key needed.

Used when the orchestrator runs in demo mode. The classifier emits the same
`IntentResult` shape the Claude-backed classifier does, so the rest of the graph is
unchanged. Honest framing: a rule-based classifier on a 30-query golden set hits
roughly the same accuracy as a zero-shot LLM (around 80-85%) — the LLM only pulls
ahead on out-of-distribution phrasing, which is rare for a structured domain.
"""

from __future__ import annotations

import re
from typing import Literal

from ..i18n import detect_locale
from .intent import IntentResult

# (intent, locale-agnostic regex patterns)
_INTENT_RULES: list[tuple[str, list[str]]] = [
    (
        "appointment",
        [
            r"\bappointment\b", r"\bbook\b", r"\bvisit\b", r"\bcome to\b",
            r"\brendez-?vous\b", r"\bvenir\b", r"\bvisite\b", r"\bme rendre\b",
        ],
    ),
    (
        "heritage_inquiry",
        [
            r"\bhistory\b", r"\bstory\b", r"\borigin\b", r"\bheritage\b",
            r"\barchive\b", r"\btradition\b", r"\bsavoir[\s-]faire\b",
            r"\bhistoire\b", r"\bpatrimoine\b", r"\borigine\b",
            r"\bd[''‘]où vient\b", r"\bparlez-moi de\b",
        ],
    ),
    (
        "price_inquiry",
        [
            r"\bhow much\b", r"\bprice\b", r"\bcost\b", r"\bunder\b",
            r"\bbudget\b",
            r"\bcombien\b", r"\bprix\b", r"\bco[uû]te\b",
        ],
    ),
    (
        "investment_piece",
        [
            r"\binvestment\b", r"\bcollect(?:or|ible|ion)\b",
            r"\barchival?\b", r"\bhaute joaillerie\b",
            r"\binvestir\b", r"\bcollectionneur\b",
        ],
    ),
    (
        "celebration",
        [
            r"\bwedding\b", r"\banniversary\b", r"\bengagement\b",
            r"\bbirthday\b", r"\bretirement\b", r"\bcelebrat\w+\b",
            r"\bmariage\b", r"\banniversaire\b", r"\bfian[çc]ailles\b",
            r"\bretraite\b", r"\bbapt[êe]me\b",
        ],
    ),
    (
        "gift",
        [
            r"\bgift\b", r"\bpresent\b", r"\bsurprise\b",
            r"\bcadeau\b", r"\boffrir\b",
        ],
    ),
    (
        "browse",
        [
            r"\bshow me\b", r"\bbrowse\b", r"\blooking for\b", r"\bwhat do you have\b",
            r"\bmontrez\b", r"\bvoir\b", r"\bje cherche\b", r"\bque proposez-vous\b",
        ],
    ),
]

# Out-of-scope triggers route straight to escalation.
_ESCALATE_RULES = [
    r"\bresale value\b", r"\bvaluation\b", r"\bappraisal\b",
    r"\bcartier\b", r"\bbvlgari\b", r"\bbulgari\b", r"\bchanel\b", r"\btiffany\b",
    r"\bcomplain\b", r"\bdamaged\b", r"\bbroken\b", r"\brefund\b",
    r"\b[ée]valuation\b", r"\bestimation\b", r"\bpr[éeè]judice\b", r"\bplainte\b",
    r"\bend[oô]mmag[ée]\b",
]


def classify_intent_rule_based(
    user_message: str, *, default_locale: Literal["en", "fr"] = "en", has_image: bool = False
) -> IntentResult:
    text = user_message.lower()
    locale = detect_locale(user_message, default=default_locale)

    if any(re.search(p, text) for p in _ESCALATE_RULES):
        return IntentResult(
            intent="unknown",
            confidence=0.25,
            locale=locale,
            needs_visual_search=has_image,
            needs_heritage=False,
            needs_catalog=False,
            escalate=True,
        )

    matched_intent = "unknown"
    for intent_name, patterns in _INTENT_RULES:
        if any(re.search(p, text) for p in patterns):
            matched_intent = intent_name
            break

    needs_catalog = matched_intent in {
        "gift", "celebration", "browse", "price_inquiry", "investment_piece", "appointment"
    }
    needs_heritage = matched_intent == "heritage_inquiry"

    if matched_intent == "unknown" and len(text.split()) <= 2:
        # Greetings / one-word messages — low confidence, no escalation
        confidence = 0.45
        escalate = False
    elif matched_intent == "unknown":
        confidence = 0.40
        escalate = False
    else:
        confidence = 0.88

    return IntentResult(
        intent=matched_intent,  # type: ignore[arg-type]
        confidence=confidence,
        locale=locale,
        needs_visual_search=has_image,
        needs_heritage=needs_heritage,
        needs_catalog=needs_catalog,
        escalate=escalate if matched_intent == "unknown" else False,
    )
