"""Recommendation / Upsell agent — ranks curated experiences by persona
affinity, and always justifies each pick.

Not an LLM: a scored ranker over a small, hand-curated list of experiences
mirroring `kb-experiences`. Each experience carries the property it belongs
to, its price, and which guest segments it suits. The score is:

    property_match (+3) + segment_match (+2) + price_tier_match (+1)

The composer surfaces the top three with a one-line why per suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .personas import Persona


@dataclass(slots=True, frozen=True)
class Experience:
    id: str
    name: str
    kb_id: str
    property: str           # "lisboa" | "algarve" | "both"
    price_eur: float
    suits_segments: tuple[str, ...]
    suits_value_bands: tuple[str, ...]   # Bronze / Silver / Gold / Platinum
    tags: tuple[str, ...]
    justification_hint: str


@dataclass(slots=True)
class ExperienceRanked:
    experience: Experience
    score: float
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.experience.id,
            "name": self.experience.name,
            "kb_id": self.experience.kb_id,
            "property": self.experience.property,
            "price_eur": self.experience.price_eur,
            "score": round(self.score, 3),
            "why": self.why,
        }


@dataclass(slots=True)
class RecommendationResult:
    ranked: list[ExperienceRanked] = field(default_factory=list)
    signals_used: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ranked": [r.as_dict() for r in self.ranked],
            "signals_used": self.signals_used,
        }


# Small, honest catalogue mirroring the KB. Not the source of truth for user-
# facing copy — that stays in kb-experiences — but the ranker needs typed
# fields the retrieval snippet can't offer.
CATALOGUE: tuple[Experience, ...] = (
    Experience(
        id="lb-fado",
        name="Fado & Supper in Alfama",
        kb_id="kb-experiences",
        property="lisboa",
        price_eur=220,
        suits_segments=("weekend_getaway", "adult_only", "extended_stay"),
        suits_value_bands=("Silver", "Gold", "Platinum"),
        tags=("cultural", "night", "adult"),
        justification_hint="a private table at a family-run casa de fado",
    ),
    Experience(
        id="lb-chiado-dawn",
        name="Chiado at Dawn",
        kb_id="kb-experiences",
        property="lisboa",
        price_eur=140,
        suits_segments=("weekend_getaway", "quick_stay", "solo_traveller",
                        "early_planner", "last_minute"),
        suits_value_bands=("Bronze", "Silver", "Gold", "Platinum"),
        tags=("cultural", "morning"),
        justification_hint="an hour walking the empty streets with a Lisboa-born urbanist",
    ),
    Experience(
        id="lb-ceramics",
        name="Ceramics Atelier",
        kb_id="kb-experiences",
        property="lisboa",
        price_eur=280,
        suits_segments=("extended_stay", "early_planner", "family_holiday"),
        suits_value_bands=("Silver", "Gold", "Platinum"),
        tags=("cultural", "craft", "family_ok"),
        justification_hint="a half-day at a working azulejo studio; you take home a hand-painted tile",
    ),
    Experience(
        id="al-sunset-sail",
        name="Sunset Sail",
        kb_id="kb-experiences",
        property="algarve",
        price_eur=480,
        suits_segments=("weekend_getaway", "premium_stay", "adult_only"),
        suits_value_bands=("Gold", "Platinum"),
        tags=("romance", "adult", "coastal"),
        justification_hint="a private catamaran along the Ponta da Piedade cliffs at sunset",
    ),
    Experience(
        id="al-kayak-dawn",
        name="Grotto Kayak at Dawn",
        kb_id="kb-experiences",
        property="algarve",
        price_eur=190,
        suits_segments=("adult_only", "early_planner", "weekend_getaway", "solo_traveller"),
        suits_value_bands=("Silver", "Gold", "Platinum"),
        tags=("active", "morning", "adult"),
        justification_hint="a guided kayak into the caves at first light, before the day-boats arrive",
    ),
    Experience(
        id="al-chef-table",
        name="Chef's Table on the Cliff",
        kb_id="kb-experiences",
        property="algarve",
        price_eur=340,
        suits_segments=("premium_stay", "adult_only", "extended_stay"),
        suits_value_bands=("Gold", "Platinum"),
        tags=("food", "adult"),
        justification_hint="a six-course tasting menu on a single table on the cliff terrace",
    ),
    Experience(
        id="both-vineyard-day",
        name="Douro / Alentejo Vineyard Day",
        kb_id="kb-experiences",
        property="both",
        price_eur=650,
        suits_segments=("extended_stay", "premium_stay", "early_planner"),
        suits_value_bands=("Gold", "Platinum"),
        tags=("wine", "day-trip"),
        justification_hint="a chauffeured private tour with a Master of Wine, three wineries, lunch included",
    ),
    Experience(
        id="both-cellar-tasting",
        name="Wine Cellar Tasting",
        kb_id="kb-experiences",
        property="both",
        price_eur=120,
        suits_segments=("last_minute", "quick_stay", "solo_traveller",
                        "weekend_getaway", "midweek_stay", "standard_stay"),
        suits_value_bands=("Bronze", "Silver", "Gold", "Platinum"),
        tags=("wine", "evening", "on_property"),
        justification_hint="90 minutes in the hotel cellar with the sommelier, four glasses",
    ),
    Experience(
        id="al-kids-club",
        name="Ondinha Kids Club Day",
        kb_id="kb-family-services",
        property="algarve",
        price_eur=70,
        suits_segments=("family_holiday",),
        suits_value_bands=("Bronze", "Silver", "Gold", "Platinum"),
        tags=("family", "day"),
        justification_hint="supervised day club for ages 4 to 11, lunch included",
    ),
)


def _property_of(persona: Persona) -> str:
    if "Lisboa" in persona.property:
        return "lisboa"
    if "Algarve" in persona.property:
        return "algarve"
    return "both"


def _why(exp: Experience, segment: str, value_band: str, prop: str) -> str:
    reasons: list[str] = []
    if exp.property == prop:
        reasons.append(f"at your property ({prop.title()})")
    elif exp.property == "both":
        reasons.append("offered at both properties")
    if segment in exp.suits_segments:
        reasons.append(f"a good fit for {segment.replace('_', ' ')} guests")
    if value_band in exp.suits_value_bands:
        reasons.append(f"priced for the {value_band} tier")
    if not reasons:
        reasons.append("a broadly-appealing option")
    return f"{exp.justification_hint} — " + "; ".join(reasons) + "."


class RecommendationAgent:
    """Scores the catalogue for a persona + snapshot and returns the top-N."""

    def __init__(self, catalogue: tuple[Experience, ...] = CATALOGUE) -> None:
        self._catalogue = catalogue

    def rank(
        self,
        persona: Persona,
        *,
        segment: str,
        value_band: str,
        top_n: int = 3,
    ) -> RecommendationResult:
        prop = _property_of(persona)
        ranked: list[ExperienceRanked] = []
        for exp in self._catalogue:
            if exp.property not in {prop, "both"}:
                continue
            score = 0.0
            if exp.property == prop:
                score += 3.0
            elif exp.property == "both":
                score += 1.5
            if segment in exp.suits_segments:
                score += 2.0
            if value_band in exp.suits_value_bands:
                score += 1.0
            ranked.append(
                ExperienceRanked(
                    experience=exp,
                    score=score,
                    why=_why(exp, segment, value_band, prop),
                )
            )
        ranked.sort(key=lambda r: r.score, reverse=True)
        return RecommendationResult(
            ranked=ranked[:top_n],
            signals_used=[
                f"property={prop}",
                f"segment={segment}",
                f"value_band={value_band}",
            ],
        )
