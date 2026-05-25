"""Conversation simulator — stress-tests the orchestrator with templated user queries.

Generates synthetic but realistic client messages from a small library of templates
(intent × locale × parameter slots), runs each through the orchestrator (offline /
demo mode by default — no API key needed), and scores the resulting (intent, retrieval,
reply) tuple on rule-based quality dimensions.

What this lets us say in an interview: "I stress-tested the system with 100 conversations
across 8 intents in two languages, without waiting for real clients to find the bugs."
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ..config import get_settings
from ..models import ClientIntent
from .groundedness import check_groundedness

Locale = Literal["en", "fr"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_OCCASIONS_EN = ["anniversary", "engagement", "wedding", "birthday", "retirement"]
_OCCASIONS_FR = ["anniversaire de mariage", "fiançailles", "mariage", "anniversaire", "retraite"]
_BUDGETS = [3_000, 5_000, 8_000, 12_000, 20_000, 50_000]
_CATEGORIES_EN = ["earrings", "necklace", "ring", "bracelet", "pendant"]
_CATEGORIES_FR = ["boucles d'oreilles", "collier", "bague", "bracelet", "pendentif"]
_MATERIALS_EN = ["yellow gold", "white gold", "rose gold"]
_MATERIALS_FR = ["or jaune", "or blanc", "or rose"]
_COLLECTIONS = ["Alhambra", "Frivole", "Perlée", "Lucky Animals", "Ballerinas"]

_TEMPLATES: list[tuple[ClientIntent, Locale, str]] = [
    (ClientIntent.GIFT, "en", "Looking for a thoughtful gift, something around CHF {budget}."),
    (ClientIntent.GIFT, "en", "A {category} for a close friend's {occasion}, budget around CHF {budget}."),
    (ClientIntent.GIFT, "fr", "Je cherche un cadeau attentionné, autour de CHF {budget}."),
    (ClientIntent.GIFT, "fr", "Un {category} pour une amie proche, environ CHF {budget}."),
    (ClientIntent.CELEBRATION, "en", "Anniversary gift for my wife, around CHF {budget}."),
    (ClientIntent.CELEBRATION, "en", "Engagement ring suggestions in {material}, around CHF {budget}."),
    (ClientIntent.CELEBRATION, "fr", "Cadeau d'{occasion} pour mon épouse, autour de CHF {budget}."),
    (ClientIntent.CELEBRATION, "fr", "Suggestions de bague de fiançailles en {material}, budget CHF {budget}."),
    (ClientIntent.HERITAGE_INQUIRY, "en", "Tell me about the history of the {collection} collection."),
    (ClientIntent.HERITAGE_INQUIRY, "en", "What is the story behind the maison's {collection}?"),
    (ClientIntent.HERITAGE_INQUIRY, "fr", "Parlez-moi de l'histoire de la collection {collection}."),
    (ClientIntent.HERITAGE_INQUIRY, "fr", "D'où vient le motif {collection} ?"),
    (ClientIntent.BROWSE, "en", "Show me {category} in {material}."),
    (ClientIntent.BROWSE, "en", "What do you have in {collection}?"),
    (ClientIntent.BROWSE, "fr", "Montrez-moi des {category} en {material}."),
    (ClientIntent.PRICE_INQUIRY, "en", "What can you show me under CHF {budget}?"),
    (ClientIntent.PRICE_INQUIRY, "en", "How much is the {collection} {category}?"),
    (ClientIntent.PRICE_INQUIRY, "fr", "Que pouvez-vous me montrer pour moins de CHF {budget} ?"),
    (ClientIntent.APPOINTMENT, "en", "I would like to book a private appointment in Geneva."),
    (ClientIntent.APPOINTMENT, "en", "Can someone come to my residence to present pieces?"),
    (ClientIntent.APPOINTMENT, "fr", "Je souhaiterais réserver un rendez-vous privé à Genève."),
    (ClientIntent.INVESTMENT_PIECE, "en", "I am building a collection of {collection} archival pieces."),
    (ClientIntent.INVESTMENT_PIECE, "en", "Looking for an investment-grade {category}."),
    (ClientIntent.INVESTMENT_PIECE, "fr", "Je constitue une collection de pièces d'archive {collection}."),
    (ClientIntent.UNKNOWN, "en", "Can you tell me the resale value of my grandmother's necklace?"),
    (ClientIntent.UNKNOWN, "en", "Do you also carry Cartier or Bvlgari?"),
    (ClientIntent.UNKNOWN, "fr", "Quelle est la valeur de revente du collier de ma grand-mère ?"),
]


def _fill(template: str, rng: random.Random, locale: Locale) -> str:
    return template.format(
        budget=rng.choice(_BUDGETS),
        occasion=rng.choice(_OCCASIONS_EN if locale == "en" else _OCCASIONS_FR),
        category=rng.choice(_CATEGORIES_EN if locale == "en" else _CATEGORIES_FR),
        material=rng.choice(_MATERIALS_EN if locale == "en" else _MATERIALS_FR),
        collection=rng.choice(_COLLECTIONS),
    )


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TurnResult:
    conversation_id: str
    timestamp: str
    template_intent: str
    detected_intent: str
    intent_match: bool
    intent_confidence: float
    locale: str
    user_message: str
    assistant_reply: str
    n_catalog_hits: int
    n_heritage_hits: int
    citation_density: float          # citations per 100 chars
    hallucination_count: int
    is_grounded: bool
    escalated: bool
    quality_score: float             # composite 0-1


def _quality_score(*, intent_match: bool, is_grounded: bool, has_evidence: bool, escalated: bool) -> float:
    score = 0.0
    score += 0.40 if intent_match else 0.0
    score += 0.30 if is_grounded else 0.0
    score += 0.20 if has_evidence else 0.0
    score += 0.10  # always award; the other three terms capture behaviour
    return round(score, 3)


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimulationReport:
    timestamp: str
    n_turns: int
    intent_accuracy: float
    grounded_rate: float
    escalation_rate: float
    avg_quality_score: float
    avg_intent_confidence: float
    per_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    per_locale: dict[str, dict[str, float]] = field(default_factory=dict)
    worst_turns: list[TurnResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_simulation(*, n_turns: int = 100, seed: int = 42) -> tuple[SimulationReport, list[TurnResult]]:
    from ..agents import Orchestrator

    rng = random.Random(seed)
    orchestrator = Orchestrator()

    samples = [rng.choice(_TEMPLATES) for _ in range(n_turns)]
    results: list[TurnResult] = []

    for template_intent, locale, template in samples:
        text = _fill(template, rng, locale)
        conv_id = f"sim-{uuid.uuid4().hex[:8]}"
        state = orchestrator.run(
            conversation_id=conv_id, user_message=text, locale=locale
        )
        catalog_hits = state.get("catalog_hits") or []
        heritage_hits = state.get("heritage_hits") or []
        reply = state.get("assistant_reply", "")
        groundedness = check_groundedness(
            reply,
            catalog_evidence=[h.piece for h in catalog_hits],
            heritage_evidence=[h.document for h in heritage_hits],
        )

        detected_intent_obj = state.get("intent", ClientIntent.UNKNOWN)
        detected_intent = detected_intent_obj.value if hasattr(detected_intent_obj, "value") else str(detected_intent_obj)
        intent_match = detected_intent == template_intent.value
        has_evidence = bool(catalog_hits) or bool(heritage_hits)
        escalated = bool(state.get("escalate"))
        citation_density = (
            (len(state.get("citations") or []) / len(reply)) * 100 if reply else 0.0
        )

        result = TurnResult(
            conversation_id=conv_id,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            template_intent=template_intent.value,
            detected_intent=detected_intent,
            intent_match=intent_match,
            intent_confidence=float(state.get("intent_confidence", 0.0)),
            locale=state.get("locale", locale),
            user_message=text,
            assistant_reply=reply,
            n_catalog_hits=len(catalog_hits),
            n_heritage_hits=len(heritage_hits),
            citation_density=round(citation_density, 3),
            hallucination_count=groundedness.hallucination_count,
            is_grounded=groundedness.is_grounded,
            escalated=escalated,
            quality_score=_quality_score(
                intent_match=intent_match,
                is_grounded=groundedness.is_grounded,
                has_evidence=has_evidence,
                escalated=escalated,
            ),
        )
        results.append(result)

    report = _aggregate(results)
    return report, results


def _aggregate(results: list[TurnResult]) -> SimulationReport:
    n = max(1, len(results))
    intent_acc = sum(1 for r in results if r.intent_match) / n
    grounded = sum(1 for r in results if r.is_grounded) / n
    escalated = sum(1 for r in results if r.escalated) / n
    avg_quality = sum(r.quality_score for r in results) / n
    avg_conf = sum(r.intent_confidence for r in results) / n

    by_intent: dict[str, list[TurnResult]] = {}
    for r in results:
        by_intent.setdefault(r.template_intent, []).append(r)
    per_intent = {
        intent: {
            "n": len(rows),
            "intent_accuracy": sum(1 for x in rows if x.intent_match) / max(1, len(rows)),
            "grounded_rate": sum(1 for x in rows if x.is_grounded) / max(1, len(rows)),
            "avg_quality": sum(x.quality_score for x in rows) / max(1, len(rows)),
        }
        for intent, rows in by_intent.items()
    }

    by_locale: dict[str, list[TurnResult]] = {}
    for r in results:
        by_locale.setdefault(r.locale, []).append(r)
    per_locale = {
        locale: {
            "n": len(rows),
            "intent_accuracy": sum(1 for x in rows if x.intent_match) / max(1, len(rows)),
            "grounded_rate": sum(1 for x in rows if x.is_grounded) / max(1, len(rows)),
            "avg_quality": sum(x.quality_score for x in rows) / max(1, len(rows)),
        }
        for locale, rows in by_locale.items()
    }

    worst = sorted(results, key=lambda r: r.quality_score)[:10]

    return SimulationReport(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        n_turns=len(results),
        intent_accuracy=intent_acc,
        grounded_rate=grounded,
        escalation_rate=escalated,
        avg_quality_score=avg_quality,
        avg_intent_confidence=avg_conf,
        per_intent=per_intent,
        per_locale=per_locale,
        worst_turns=worst,
    )


def write_simulation_report(report: SimulationReport, turns: list[TurnResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": report.timestamp,
        "n_turns": report.n_turns,
        "intent_accuracy": report.intent_accuracy,
        "grounded_rate": report.grounded_rate,
        "escalation_rate": report.escalation_rate,
        "avg_quality_score": report.avg_quality_score,
        "avg_intent_confidence": report.avg_intent_confidence,
        "per_intent": report.per_intent,
        "per_locale": report.per_locale,
        "worst_turns": [asdict(t) for t in report.worst_turns],
        "turns": [asdict(t) for t in turns],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_report_path() -> Path:
    return get_settings().data_dir / "eval" / "simulation_report.json"
