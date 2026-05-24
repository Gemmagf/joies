"""Client profile model and segment vocabulary."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, NonNegativeFloat

from .catalog import Collection


class ClientSegment(StrEnum):
    """Four-tier segmentation used to tailor concierge behaviour and prioritise SA follow-up."""

    PROSPECT = "prospect"          # < 1y tenure, low spend, exploratory
    ESTABLISHED = "established"    # repeat client, predictable basket
    VIP = "vip"                    # high spend, multiple collections, private-viewing eligible
    COLLECTOR = "collector"        # archive-driven, high jewelry interest


class ClientProfile(BaseModel):
    id: str = Field(pattern=r"^CL-[A-Z0-9-]+$")
    display_name: str
    segment: ClientSegment
    tenure_years: NonNegativeFloat
    lifetime_spend_chf: NonNegativeFloat
    collections_owned: list[Collection] = Field(default_factory=list)
    preferred_language: Literal["en", "fr"] = "en"
    city: str | None = None
    last_appointment_at: datetime | None = None
    notes: str | None = None

    def days_since_last_appointment(self, *, now: datetime | None = None) -> int | None:
        if self.last_appointment_at is None:
            return None
        ref = now or datetime.now(self.last_appointment_at.tzinfo)
        return max(0, (ref - self.last_appointment_at).days)
