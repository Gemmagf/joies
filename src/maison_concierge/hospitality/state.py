"""LangGraph state for the hospitality orchestrator."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from .intent import HospitalityIntent
from .kb import KBSearchResult
from .profile import ProfileSnapshot


def _take_last(_: object, new: object) -> object:
    return new


def _merge_dicts(current: dict | None, new: dict | None) -> dict:
    merged = dict(current or {})
    merged.update(new or {})
    return merged


class HospitalityState(TypedDict, total=False):
    conversation_id: str
    user_message: str
    persona_id: str | None
    locale: str

    intent: HospitalityIntent
    intent_confidence: float
    route_profile: bool
    route_knowledge: bool
    route_booking: bool
    route_recommendation: bool
    escalate: bool
    escalation_reason: str | None

    profile_snapshot: ProfileSnapshot | None
    kb_hits: list[KBSearchResult]

    assistant_reply: Annotated[str, _take_last]
    citations: list[str]
    trace: Annotated[dict[str, Any], _merge_dicts]
