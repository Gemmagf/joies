"""Composer — turns intent + retrieval + profile + booking + recommendation
into a reply.

Deterministic templating for the MVP. Every route has its own branch: the
compose node stays pure, and any user-facing prose lives here. The
guardrail node runs after and may still rewrite the reply.
"""

from __future__ import annotations

from dataclasses import dataclass

from .booking import BookingActionResult
from .intent import HospitalityIntent
from .kb import KBSearchResult
from .personas import Persona
from .profile import ProfileSnapshot
from .recommend import RecommendationResult

_ESCALATION_REPLY = (
    "I'm going to bring this to a human colleague right away. A member of the "
    "Concierge d'Honneur will follow up within a few minutes. If it's urgent, "
    "please call the property directly at +351 21 000 1234."
)

_UNKNOWN_REPLY = (
    "I'm not sure I've understood — could you rephrase? I can help with your "
    "guest profile, property policies, dining, spa, family services, and "
    "curated experiences."
)

_GREETING_REPLIES = (
    "Welcome, {name}. I'm your concierge for {property}. How may I help — "
    "your booking, an experience, or something about the property?",
    "Hello {name} — delighted to have you. Ask me anything about your stay "
    "at {property}, or about your guest profile.",
    "Good to hear from you, {name}. Anything you'd like me to look up "
    "about {property}, your booking, or an experience for your stay?",
)


@dataclass(slots=True)
class ComposerOutput:
    reply: str
    citations: list[str]
    mode: str  # "templated" | "claude"


def _compose_profile(profile: ProfileSnapshot | None, persona: Persona | None) -> ComposerOutput:
    if profile is None:
        return ComposerOutput(
            reply=(
                "I'd need to know which guest you're asking about — please pick "
                "a persona from the side panel."
            ),
            citations=[],
            mode="templated",
        )
    greeting = f"{persona.display_name}, " if persona else ""
    notes = " ".join(profile.notes)
    return ComposerOutput(
        reply=(
            f"{greeting}here's the snapshot our analytics returned:\n\n"
            f"- {profile.summary()}\n\n{notes}"
        ),
        citations=[],
        mode="templated",
    )


def _compose_knowledge(kb_hits: list[KBSearchResult]) -> ComposerOutput:
    if not kb_hits:
        return ComposerOutput(
            reply=(
                "I couldn't find that in our property notes — I'll escalate to a "
                "human colleague so we can answer properly."
            ),
            citations=[],
            mode="templated",
        )
    best = kb_hits[0]
    lines = [
        f"From our *{best.page.title}* notes:",
        "",
        best.snippet,
        "",
        f"[Source: {best.page.id}]",
    ]
    if len(kb_hits) > 1:
        others = ", ".join(f"{h.page.id}" for h in kb_hits[1:])
        lines.append(f"Related notes you might also want: {others}.")
    return ComposerOutput(
        reply="\n".join(lines),
        citations=[h.page.id for h in kb_hits],
        mode="templated",
    )


def _compose_booking(result: BookingActionResult | None) -> ComposerOutput:
    if result is None:
        return ComposerOutput(
            reply=(
                "I'd need to know which guest you're asking about — please pick "
                "a persona from the side panel."
            ),
            citations=[],
            mode="templated",
        )
    return ComposerOutput(reply=result.message, citations=[], mode="templated")


def _compose_recommendation(
    rec: RecommendationResult | None, persona: Persona | None
) -> ComposerOutput:
    if rec is None or not rec.ranked:
        return ComposerOutput(
            reply=(
                "I couldn't rank experiences without knowing your profile — please "
                "pick a persona from the side panel."
            ),
            citations=[],
            mode="templated",
        )
    greeting = f"{persona.display_name}, " if persona else ""
    signals = ", ".join(rec.signals_used)
    lines = [
        f"{greeting}three experiences I'd propose, ranked by fit "
        f"({signals}):",
        "",
    ]
    seen_kb: set[str] = set()
    for i, r in enumerate(rec.ranked, start=1):
        lines.append(
            f"{i}. **{r.experience.name}** (€{r.experience.price_eur:.0f}) "
            f"— {r.why}"
        )
        seen_kb.add(r.experience.kb_id)
    lines.append("")
    lines.append(f"[Source: {', '.join(sorted(seen_kb))}]")
    return ComposerOutput(
        reply="\n".join(lines),
        citations=sorted(seen_kb),
        mode="templated",
    )


def _compose_greeting(persona: Persona | None) -> ComposerOutput:
    """Pick a deterministic warm reply — we index into the pool by persona id
    so the same guest gets the same greeting, but different guests differ."""
    name = persona.display_name.split()[0] if persona else "there"
    property_name = persona.property if persona else "the Maison"
    if persona is None:
        template = _GREETING_REPLIES[0]
    else:
        # Deterministic pick from the pool — sum of ord() over id is stable.
        idx = sum(ord(c) for c in persona.id) % len(_GREETING_REPLIES)
        template = _GREETING_REPLIES[idx]
    return ComposerOutput(
        reply=template.format(name=name, property=property_name),
        citations=[],
        mode="templated",
    )


def compose_templated(
    *,
    intent: HospitalityIntent,
    profile: ProfileSnapshot | None,
    kb_hits: list[KBSearchResult],
    booking_result: BookingActionResult | None = None,
    recommendation: RecommendationResult | None = None,
    persona: Persona | None = None,
    escalate: bool = False,
) -> ComposerOutput:
    if escalate or intent == HospitalityIntent.ESCALATE:
        return ComposerOutput(reply=_ESCALATION_REPLY, citations=[], mode="templated")
    if intent == HospitalityIntent.GREETING:
        return _compose_greeting(persona)
    if intent == HospitalityIntent.PROFILE:
        return _compose_profile(profile, persona)
    if intent == HospitalityIntent.KNOWLEDGE:
        return _compose_knowledge(kb_hits)
    if intent == HospitalityIntent.BOOKING:
        return _compose_booking(booking_result)
    if intent == HospitalityIntent.RECOMMENDATION:
        return _compose_recommendation(recommendation, persona)
    return ComposerOutput(reply=_UNKNOWN_REPLY, citations=[], mode="templated")
