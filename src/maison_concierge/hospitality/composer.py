"""Composer — turns intent + retrieval + profile snapshot into a reply.

Two modes:
- `compose_templated` — deterministic, always available. Used in demo mode
  and by the eval harness (so groundedness assertions don't fight the model).
- `compose_with_claude` — uses the Anthropic Messages API with prompt
  caching on the maison's brand system prompt. Used when a real API key is
  set; falls back to templated on any error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .intent import HospitalityIntent
from .kb import KBSearchResult
from .personas import Persona
from .profile import ProfileSnapshot

_ESCALATION_REPLY = (
    "I'm going to bring this to a human colleague right away. A member of the "
    "Concierge d'Honneur will follow up within a few minutes. If it's urgent, "
    "please call the property directly at +351 21 000 1234."
)

_STUB_BOOKING_REPLY = (
    "Booking changes are handled by our reservations team — I can hand off "
    "with your booking reference. In the next release I'll be able to change "
    "dates and room types directly."
)

_STUB_RECOMMENDATION_REPLY = (
    "I'll be able to recommend curated experiences from your next stay "
    "shortly — that agent is next on the build. For now, our full experience "
    "catalogue is in kb-experiences."
)

_UNKNOWN_REPLY = (
    "I'm not sure I've understood — could you rephrase? I can help with your "
    "guest profile, property policies, dining, spa, family services, and "
    "curated experiences."
)


@dataclass(slots=True)
class ComposerOutput:
    reply: str
    citations: list[str]
    mode: str  # "templated" | "claude"


def compose_templated(
    *,
    intent: HospitalityIntent,
    profile: ProfileSnapshot | None,
    kb_hits: list[KBSearchResult],
    persona: Persona | None = None,
    escalate: bool = False,
) -> ComposerOutput:
    if escalate or intent == HospitalityIntent.ESCALATE:
        return ComposerOutput(reply=_ESCALATION_REPLY, citations=[], mode="templated")

    if intent == HospitalityIntent.PROFILE:
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
            reply=f"{greeting}here's the snapshot our analytics returned:\n\n"
                  f"- {profile.summary()}\n\n{notes}",
            citations=[],
            mode="templated",
        )

    if intent == HospitalityIntent.KNOWLEDGE:
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

    if intent == HospitalityIntent.BOOKING:
        return ComposerOutput(reply=_STUB_BOOKING_REPLY, citations=[], mode="templated")

    if intent == HospitalityIntent.RECOMMENDATION:
        return ComposerOutput(
            reply=_STUB_RECOMMENDATION_REPLY, citations=[], mode="templated"
        )

    return ComposerOutput(reply=_UNKNOWN_REPLY, citations=[], mode="templated")
