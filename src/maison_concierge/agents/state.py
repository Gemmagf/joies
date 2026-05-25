"""LangGraph state schema for the orchestrator."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from ..models import Citation, ClientIntent, EscalationReason
from ..retrieval import CatalogSearchResult, HeritageSearchResult, VisualSearchResult

Locale = Literal["en", "fr"]


def _take_last(_: object, new: object) -> object:
    return new


def _merge_dicts(current: dict | None, new: dict | None) -> dict:
    merged = dict(current or {})
    merged.update(new or {})
    return merged


class OrchestratorState(TypedDict, total=False):
    conversation_id: str
    locale: Locale
    user_message: str
    image_bytes: bytes | None

    intent: ClientIntent
    intent_confidence: float
    route_catalog: bool
    route_heritage: bool
    route_visual: bool
    escalate: bool
    escalation_reason: EscalationReason | None

    catalog_hits: list[CatalogSearchResult]
    heritage_hits: list[HeritageSearchResult]
    visual_hits: list[VisualSearchResult]

    assistant_reply: Annotated[str, _take_last]
    citations: list[Citation]
    trace: Annotated[dict[str, Any], _merge_dicts]
