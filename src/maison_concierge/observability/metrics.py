"""Append-only event log + aggregated snapshots for the stakeholder dashboard.

Persists to JSONL on disk under `metrics_dir/events.jsonl`. The dashboard reads the file
directly so the demo can render even with the chat app stopped.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import ClientIntent, EscalationReason

EventType = Literal[
    "conversation_started",
    "intent_detected",
    "tool_called",
    "escalation",
    "appointment_booked",
    "lead_flagged",
    "rating_submitted",
]


class Event(BaseModel):
    type: EventType
    conversation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    total_conversations: int
    escalation_rate: float
    intents: dict[str, int]
    escalation_reasons: dict[str, int]
    avg_rating: float | None
    appointments_booked: int
    leads_by_tier: dict[str, int]


class MetricsRecorder:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: Event) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def record_intent(
        self, conversation_id: str, intent: ClientIntent, confidence: float
    ) -> None:
        self.record(
            Event(
                type="intent_detected",
                conversation_id=conversation_id,
                payload={"intent": intent.value, "confidence": confidence},
            )
        )

    def record_escalation(
        self, conversation_id: str, reason: EscalationReason
    ) -> None:
        self.record(
            Event(
                type="escalation",
                conversation_id=conversation_id,
                payload={"reason": reason.value},
            )
        )

    def record_tool(self, conversation_id: str, tool: str, ok: bool) -> None:
        self.record(
            Event(
                type="tool_called",
                conversation_id=conversation_id,
                payload={"tool": tool, "ok": ok},
            )
        )

    def record_rating(self, conversation_id: str, rating: int) -> None:
        self.record(
            Event(
                type="rating_submitted",
                conversation_id=conversation_id,
                payload={"rating": rating},
            )
        )

    def load_events(self) -> list[Event]:
        if not self._path.exists():
            return []
        events: list[Event] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(Event.model_validate_json(line))
        return events

    def snapshot(self) -> MetricsSnapshot:
        events = self.load_events()
        conversations = {e.conversation_id for e in events if e.type == "conversation_started"}
        intents = Counter(
            e.payload.get("intent", "unknown")
            for e in events
            if e.type == "intent_detected"
        )
        escalations = [e for e in events if e.type == "escalation"]
        escalation_reasons = Counter(
            e.payload.get("reason", "unknown") for e in escalations
        )
        ratings = [
            int(e.payload["rating"])
            for e in events
            if e.type == "rating_submitted" and "rating" in e.payload
        ]
        appointments = sum(1 for e in events if e.type == "appointment_booked")
        leads = Counter(
            e.payload.get("tier", "watch") for e in events if e.type == "lead_flagged"
        )
        total = max(len(conversations), 1)
        return MetricsSnapshot(
            total_conversations=len(conversations),
            escalation_rate=len(escalations) / total,
            intents=dict(intents),
            escalation_reasons=dict(escalation_reasons),
            avg_rating=(sum(ratings) / len(ratings)) if ratings else None,
            appointments_booked=appointments,
            leads_by_tier=dict(leads),
        )


@lru_cache(maxsize=1)
def get_recorder() -> MetricsRecorder:
    settings = get_settings()
    return MetricsRecorder(settings.metrics_dir / "events.jsonl")
