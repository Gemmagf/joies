"""Tool I/O schemas — appointment booking, pricing, lead flagging."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

AppointmentChannel = Literal["boutique", "video_call", "private_residence"]


class AppointmentRequest(BaseModel):
    client_name: str
    client_email: EmailStr | None = None
    locale: Literal["en", "fr"] = "en"
    boutique: str = Field(default="Geneva — Rue du Rhône")
    channel: AppointmentChannel = "boutique"
    preferred_at: datetime
    notes: str | None = None
    piece_ids_of_interest: list[str] = Field(default_factory=list)


class Appointment(BaseModel):
    reference: str
    request: AppointmentRequest
    confirmed_at: datetime
    advisor_assigned: str
