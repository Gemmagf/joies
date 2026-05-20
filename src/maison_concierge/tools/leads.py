"""Lead flagging — marks conversations with HNW signals for a human advisor follow-up."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings

LeadTier = Literal["watch", "warm", "hot"]

_HNW_TERMS = (
    "investment", "private viewing", "high jewelry", "haute joaillerie",
    "anniversary", "collector", "archive", "private residence", "discrétion",
)


class LeadFlag(BaseModel):
    conversation_id: str
    client_hint: str | None = None
    tier: LeadTier
    reason: str
    flagged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _leads_path() -> Path:
    return get_settings().data_dir / "leads.jsonl"


def _tier_for(text: str) -> tuple[LeadTier, str]:
    lowered = text.lower()
    matches = [term for term in _HNW_TERMS if term in lowered]
    if not matches:
        return "watch", "No explicit HNW signal."
    if len(matches) >= 3:
        return "hot", f"Multiple HNW signals: {', '.join(matches[:3])}."
    return "warm", f"HNW signal: {', '.join(matches)}."


def flag_high_value_lead(
    conversation_id: str, transcript_text: str, *, client_hint: str | None = None
) -> LeadFlag:
    tier, reason = _tier_for(transcript_text)
    flag = LeadFlag(
        conversation_id=conversation_id,
        client_hint=client_hint,
        tier=tier,
        reason=reason,
    )
    path = _leads_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(flag.model_dump_json() + "\n")
    return flag
