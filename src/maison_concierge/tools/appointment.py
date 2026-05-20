"""Mock appointment booking tool.

The real implementation would call the maison's CRM (Salesforce / bespoke). The mock
generates a deterministic reference and writes the booking to data/appointments.jsonl so
the dashboard can show real demo activity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from ..models import Appointment, AppointmentRequest

_ADVISORS = [
    "Camille Laurent — Geneva",
    "Élodie Garnier — Paris",
    "Sophie Marchetti — Geneva",
    "Antoine Rey — Paris",
]


def _assign_advisor(client_name: str) -> str:
    h = int(hashlib.sha256(client_name.encode("utf-8")).hexdigest(), 16)
    return _ADVISORS[h % len(_ADVISORS)]


def _reference(request: AppointmentRequest) -> str:
    payload = f"{request.client_name}|{request.preferred_at.isoformat()}|{request.boutique}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8].upper()
    return f"VCA-APT-{digest}"


def _appointments_path() -> Path:
    return get_settings().data_dir / "appointments.jsonl"


def book_appointment(request: AppointmentRequest) -> Appointment:
    appointment = Appointment(
        reference=_reference(request),
        request=request,
        confirmed_at=datetime.now(timezone.utc),
        advisor_assigned=_assign_advisor(request.client_name),
    )
    path = _appointments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(appointment.model_dump_json() + "\n")
    return appointment


class AppointmentTool:
    """Adapter exposing the Anthropic tool-use schema."""

    name = "book_appointment"
    description = (
        "Book a private appointment for the client. Use when the client requests an "
        "appointment, when escalating a high-value lead, or when the client wants to view "
        "a specific piece in person."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "client_name": {"type": "string"},
            "client_email": {"type": "string"},
            "locale": {"type": "string", "enum": ["en", "fr"]},
            "boutique": {"type": "string"},
            "channel": {
                "type": "string",
                "enum": ["boutique", "video_call", "private_residence"],
            },
            "preferred_at": {"type": "string", "format": "date-time"},
            "notes": {"type": "string"},
            "piece_ids_of_interest": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["client_name", "preferred_at"],
    }

    def __call__(self, **kwargs: object) -> Appointment:
        request = AppointmentRequest.model_validate(kwargs)
        return book_appointment(request)
