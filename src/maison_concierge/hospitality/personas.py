"""Synthetic guest personas — real distributions, invented identities.

We take deterministic sample rows from the Antonio et al 2019 dataset and
attach a synthetic name / email / booking reference. The rows themselves —
lead time, deposit type, party size, ADR, arrival date — are real. The
identities are not tied to any real person.

`generate` produces the JSON. `load_personas` reads it. The Streamlit UI
picks from these and the Profile agent scores them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import get_settings

PERSONAS_PATH_DEFAULT = "personas/hospitality_personas.json"


@dataclass(slots=True, frozen=True)
class Persona:
    id: str                    # PER-0001
    display_name: str
    email: str
    nationality: str           # 3-letter country code from dataset
    property: str              # "Maison Lisboa" | "Maison Algarve"
    booking_ref: str           # e.g. LM-2025-0042
    arrival_date: date
    departure_date: date
    lead_time_days: int
    party_size: int
    is_repeated_guest: bool
    previous_cancellations: int
    booking: dict[str, Any] = field(default_factory=dict)  # raw feature dict

    @property
    def stay_summary(self) -> str:
        nights = (self.departure_date - self.arrival_date).days
        who = f"{self.party_size} guest{'s' if self.party_size != 1 else ''}"
        return (
            f"{self.property} — {who}, "
            f"{nights} night{'s' if nights != 1 else ''}, "
            f"arriving {self.arrival_date.isoformat()}"
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arrival_date"] = self.arrival_date.isoformat()
        d["departure_date"] = self.departure_date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Persona:
        return cls(
            id=d["id"],
            display_name=d["display_name"],
            email=d["email"],
            nationality=d["nationality"],
            property=d["property"],
            booking_ref=d["booking_ref"],
            arrival_date=date.fromisoformat(d["arrival_date"]),
            departure_date=date.fromisoformat(d["departure_date"]),
            lead_time_days=int(d["lead_time_days"]),
            party_size=int(d["party_size"]),
            is_repeated_guest=bool(d["is_repeated_guest"]),
            previous_cancellations=int(d["previous_cancellations"]),
            booking=dict(d.get("booking", {})),
        )


def personas_path() -> Path:
    return get_settings().data_dir / PERSONAS_PATH_DEFAULT


@lru_cache(maxsize=1)
def load_personas() -> list[Persona]:
    path = personas_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Persona.from_dict(item) for item in raw]


def persona_by_id(persona_id: str) -> Persona | None:
    for p in load_personas():
        if p.id == persona_id:
            return p
    return None


def save_personas(personas: list[Persona]) -> Path:
    path = personas_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.as_dict() for p in personas]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_personas.cache_clear()
    return path
