"""Intent classification for the hospitality concierge.

Two implementations:
- rule-based (deterministic, always available, used in demo mode and eval)
- Claude-based (added in a later phase)

The rule-based classifier is intentionally simple keyword lookup — it is the
baseline the eval harness holds every route against, and it is fast enough
that the demo Streamlit sidebar never blocks on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class HospitalityIntent(StrEnum):
    PROFILE = "profile"          # "am I likely to cancel", "what's my segment"
    KNOWLEDGE = "knowledge"      # policy / property / amenity questions
    BOOKING = "booking"          # dates, changes, cancellation actions
    RECOMMENDATION = "recommendation"  # "what should I do", "suggest"
    ESCALATE = "escalate"        # complaint, harm, out-of-scope
    UNKNOWN = "unknown"


@dataclass(slots=True)
class IntentResult:
    intent: HospitalityIntent
    confidence: float
    matched_terms: list[str]
    escalate: bool
    reason: str | None = None


_PROFILE_TERMS = (
    r"my (risk|profile|segment|value|status|tier|band)",
    r"am i (likely|going) to (cancel|churn)",
    r"cancel(l|)ation risk",
    r"no[- ]?show risk",
    r"(value|clv) (band|tier|score)",
    r"(what|which) (value )?band am i in",
    r"how valuable",
    r"(vip|loyalty|repeat guest)",
    r"am i a repeat",
    r"who am i",
    r"remind me (who i am|of my)",
)

_KNOWLEDGE_TERMS = (
    r"(policy|policies|rule|rules)",
    r"(check[- ]?in|check[- ]?out|arrival|departure)",
    r"(spa|pool|hammam|treatment|wellness)",
    r"(restaurant|dining|breakfast|dinner|menu|dress code|reservation)",
    r"(cancel|refund|deposit)(?! my)",
    r"(family|kids|children|crib|babysit|kids club)",
    r"(pet|dog|cat)",
    r"(accessibility|wheelchair|mobility|allergen|gluten|vegan|kosher)",
    r"(sustainab|green key|energy|water)",
    r"(what time|opening hours|hours|when does)",
    r"(where is|how do i find|address)",
    r"(airport|transfer|taxi|chauffeur|shuttle|helicopter)",
    r"(wifi|internet|password)",
    r"(dress code|smart casual|jacket required)",
    r"(experience|activity|excursion|tour|fado|sailing|kayak)",
)

_BOOKING_TERMS = (
    r"(change|modify|move|extend|shorten) (my|the) (booking|reservation|dates|stay)",
    r"(cancel|book) (my|the|a) (booking|reservation|stay|room)",
    r"(add|upgrade) (a|my) room",
    r"upgrade my (room|suite|stay)",
    r"(when|what date) (do i|am i) arriv",
    r"my (arrival|departure|check[- ]?in) date",
)

_RECOMMENDATION_TERMS = (
    r"\b(recommend|suggest|propose)\b",
    r"what should i do",
    r"(what to do|what's on|what's happening) (tomorrow|tonight|today|this weekend)",
    r"(best|top|favourite) (restaurant|experience|dish|spot)",
    r"(surprise me|delight me)",
)

_ESCALATE_TERMS = (
    r"(complain|complaint|awful|terrible|angry|furious|refund now|manager)",
    r"(unsafe|dangerous|hurt|injur|medical emergency|police)",
    r"(discriminat|racist|sexist|harass)",
    # PII / off-limits requests about other people.
    r"(other|another|else|different) (guest|person|customer|client)",
    r"(credit card|social security|passport|home address|phone number).*(number|of)",
    r"someone else's",
    # "What is <Name>'s (home) address/phone/etc" — asking about a named person.
    r"\w+'s (home |personal |private )?(address|phone|email|credit card|passport|id number)",
    # Adversarial / prompt-injection.
    r"ignore (all )?(previous|prior|the) instructions",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"(write|create|make|draft|post) .{0,40}fake .{0,40}(review|testimonial|complaint|rating|feedback)",
    r"fabricate",
)


def _score(text: str, patterns: tuple[str, ...]) -> tuple[int, list[str]]:
    matches: list[str] = []
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            matches.append(m.group(0))
    return len(matches), matches


def classify_intent_rule_based(
    text: str, *, default_locale: str = "en"
) -> IntentResult:
    """Rule-based classifier.

    Priority order (a match in a higher tier wins outright, even against a
    higher raw count downstream):

      escalate  >  booking-action  >  profile  >  recommendation  >  knowledge

    Rationale: verbs and self-references are stronger intent signals than
    topical nouns. A message like "suggest an activity for our kids" carries
    knowledge-flavour nouns (kids, activity) *and* a recommendation verb
    (suggest) — the verb should win, because the user is asking for a
    suggestion, not for the kids-club policy.

    Returns UNKNOWN with confidence 0 when nothing matches.
    """
    _ = default_locale
    lowered = text.lower().strip()

    for intent, terms, base_conf in (
        (HospitalityIntent.ESCALATE, _ESCALATE_TERMS, 0.70),
        (HospitalityIntent.BOOKING, _BOOKING_TERMS, 0.65),
        (HospitalityIntent.PROFILE, _PROFILE_TERMS, 0.60),
        (HospitalityIntent.RECOMMENDATION, _RECOMMENDATION_TERMS, 0.60),
        (HospitalityIntent.KNOWLEDGE, _KNOWLEDGE_TERMS, 0.55),
    ):
        count, matched = _score(lowered, terms)
        if count > 0:
            confidence = min(0.95, base_conf + 0.10 * count)
            is_escalation = intent == HospitalityIntent.ESCALATE
            return IntentResult(
                intent=intent,
                confidence=confidence,
                matched_terms=matched,
                escalate=is_escalation,
                reason="explicit escalation term" if is_escalation else None,
            )

    return IntentResult(
        intent=HospitalityIntent.UNKNOWN,
        confidence=0.0,
        matched_terms=[],
        escalate=False,
    )
