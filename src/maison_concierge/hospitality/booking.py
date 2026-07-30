"""Booking & Itinerary agent — deterministic tool calls against a per-session
booking store.

The LLM never picks refund amounts, dates, or capacity. Those come from
pure-Python tools that read the persona's booking, apply the cancellation
policy exactly as written in `kb-cancellation`, and return a structured
result the composer templates into a reply.

Actions:
- CANCEL       — deterministic refund calc per deposit_type
- EXTEND       — +N nights (default 1), capacity trivially assumed
- CHANGE_DATES — recognised, but parameter extraction is deferred:
                  the composer asks the guest to confirm a date
- UPGRADE      — always requires human approval (pricing rules)
- INFO         — "what date do I arrive on" — read-only summary
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, ClassVar

from .personas import Persona


class BookingAction(StrEnum):
    CANCEL = "cancel"
    EXTEND = "extend"
    CHANGE_DATES = "change_dates"
    UPGRADE = "upgrade"
    INFO = "info"


@dataclass(slots=True)
class BookingActionResult:
    action: BookingAction
    ok: bool                 # tool applied without deferring to human
    booking_ref: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] | None = None
    refund_eur: float | None = None
    message: str = ""
    requires_human: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


_CANCEL_RE = re.compile(r"\bcancel\b", re.IGNORECASE)
_EXTEND_RE = re.compile(
    r"(extend|another night|extra night|stay .* longer|one more night|two more nights)",
    re.IGNORECASE,
)
_CHANGE_DATES_RE = re.compile(
    r"(change|move|modify|shift) (my|the) (booking|reservation|dates|stay)",
    re.IGNORECASE,
)
_UPGRADE_RE = re.compile(r"\bupgrade\b", re.IGNORECASE)
_INFO_RE = re.compile(
    r"(what date|when) (do i|am i) (arriv|leav|check[- ]?in|check[- ]?out)",
    re.IGNORECASE,
)


def classify_booking_action(text: str) -> BookingAction:
    """Sub-classifier inside the BOOKING intent. Order matters: cancel wins
    over extend if both appear (e.g., 'cancel my extension')."""
    lowered = text.lower()
    if _INFO_RE.search(lowered):
        return BookingAction.INFO
    if _CANCEL_RE.search(lowered):
        return BookingAction.CANCEL
    if _UPGRADE_RE.search(lowered):
        return BookingAction.UPGRADE
    if _EXTEND_RE.search(lowered):
        return BookingAction.EXTEND
    if _CHANGE_DATES_RE.search(lowered):
        return BookingAction.CHANGE_DATES
    return BookingAction.INFO


def _stay_summary(persona: Persona) -> dict[str, Any]:
    nights = (persona.departure_date - persona.arrival_date).days
    return {
        "booking_ref": persona.booking_ref,
        "property": persona.property,
        "arrival_date": persona.arrival_date.isoformat(),
        "departure_date": persona.departure_date.isoformat(),
        "nights": nights,
        "party_size": persona.party_size,
        "deposit_type": persona.booking.get("deposit_type", "unknown"),
    }


def _refund_fraction(deposit_type: str, days_until_arrival: int) -> float:
    """Return the fraction of the deposit that would be refunded on cancel,
    per the policy in `kb-cancellation`."""
    d = (deposit_type or "").strip().lower()
    if d == "non refund":
        return 0.0
    if d == "refundable":
        if days_until_arrival >= 7:
            return 1.0
        if days_until_arrival >= 2:
            return 0.0  # deposit retained, rest not charged
        return 0.0
    # No deposit: no refund concept
    if d in {"no deposit", ""}:
        if days_until_arrival >= 2:
            return 1.0  # nothing charged
        return 0.0  # first-night rule
    return 0.0


class BookingAgent:
    """Session-scoped view over the persona's booking. All mutations are
    kept in-memory so multiple guests can play with the demo simultaneously
    without stepping on each other."""

    def __init__(self, today: date | None = None) -> None:
        self._today = today or date.today()
        self._staged: dict[str, dict[str, Any]] = {}

    def _live_booking(self, persona: Persona) -> dict[str, Any]:
        return self._staged.setdefault(persona.id, _stay_summary(persona))

    def handle(self, persona: Persona, user_message: str) -> BookingActionResult:
        action = classify_booking_action(user_message)
        if action == BookingAction.INFO:
            return self._info(persona)
        if action == BookingAction.CANCEL:
            return self._cancel(persona)
        if action == BookingAction.EXTEND:
            return self._extend(persona, extra_nights=self._parse_extra_nights(user_message))
        if action == BookingAction.CHANGE_DATES:
            return self._change_dates(persona)
        if action == BookingAction.UPGRADE:
            return self._upgrade(persona)
        return self._info(persona)

    # ── individual tools ─────────────────────────────────────────────
    def _info(self, persona: Persona) -> BookingActionResult:
        state = self._live_booking(persona)
        return BookingActionResult(
            action=BookingAction.INFO,
            ok=True,
            booking_ref=state["booking_ref"],
            before=state,
            after=state,
            message=(
                f"Your booking {state['booking_ref']} is at {state['property']}: "
                f"arriving {state['arrival_date']}, departing {state['departure_date']} "
                f"({state['nights']} night(s), party of {state['party_size']})."
            ),
        )

    def _cancel(self, persona: Persona) -> BookingActionResult:
        before = self._live_booking(persona)
        days = max(
            0,
            (persona.arrival_date - self._today).days,
        )
        fraction = _refund_fraction(before["deposit_type"], days)
        after = {**before, "status": "cancelled", "refund_fraction": fraction}
        self._staged[persona.id] = after

        # Deposit isn't in the dataset; ADR × nights is a reasonable stand-in.
        adr = float(persona.booking.get("adr", 0.0) or 0.0)
        stay_value = adr * before["nights"]
        refund_eur = round(stay_value * fraction, 2)

        deposit_note = {
            "non refund": "This is a non-refundable rate, so the deposit is retained per policy.",
            "refundable": (
                "This is a refundable-deposit rate."
                if fraction > 0
                else "The deposit is retained because we're inside the 7-day window."
            ),
            "no deposit": "This is a flexible rate — nothing is charged when cancelling outside 48h.",
        }.get(before["deposit_type"].lower(), "")

        return BookingActionResult(
            action=BookingAction.CANCEL,
            ok=True,
            booking_ref=before["booking_ref"],
            before=before,
            after=after,
            refund_eur=refund_eur,
            message=(
                f"Cancelled {before['booking_ref']}. {deposit_note} "
                f"Refund on this stay: €{refund_eur:.0f} "
                f"({int(fraction * 100)}% of ADR × nights). "
                "You will receive a written confirmation within 30 minutes."
            ),
        )

    def _extend(self, persona: Persona, *, extra_nights: int) -> BookingActionResult:
        before = self._live_booking(persona)
        new_departure = (
            date.fromisoformat(before["departure_date"]) + timedelta(days=extra_nights)
        )
        after = {
            **before,
            "departure_date": new_departure.isoformat(),
            "nights": before["nights"] + extra_nights,
        }
        self._staged[persona.id] = after
        return BookingActionResult(
            action=BookingAction.EXTEND,
            ok=True,
            booking_ref=before["booking_ref"],
            before=before,
            after=after,
            message=(
                f"Extended {before['booking_ref']} by {extra_nights} night(s). "
                f"New departure: {after['departure_date']} "
                f"({after['nights']} nights total)."
            ),
        )

    def _change_dates(self, persona: Persona) -> BookingActionResult:
        before = self._live_booking(persona)
        return BookingActionResult(
            action=BookingAction.CHANGE_DATES,
            ok=False,
            booking_ref=before["booking_ref"],
            before=before,
            after=None,
            requires_human=False,
            message=(
                f"Happy to move {before['booking_ref']} — please share the "
                "new arrival date in YYYY-MM-DD format and I'll confirm the "
                "same length of stay. If the length should change too, tell "
                "me both dates."
            ),
        )

    def _upgrade(self, persona: Persona) -> BookingActionResult:
        before = self._live_booking(persona)
        return BookingActionResult(
            action=BookingAction.UPGRADE,
            ok=False,
            booking_ref=before["booking_ref"],
            before=before,
            after=None,
            requires_human=True,
            message=(
                f"Upgrade requests on {before['booking_ref']} go through a "
                "human colleague — pricing depends on category availability "
                "on your dates. I've flagged the request; you'll hear back "
                "within an hour."
            ),
        )

    # ── helpers ──────────────────────────────────────────────────────
    _WORD_TO_INT: ClassVar[dict[str, int]] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "another": 1, "an": 1, "a": 1,
    }

    @classmethod
    def _parse_extra_nights(cls, text: str) -> int:
        lowered = text.lower()
        m = re.search(r"(\d+)\s*(more\s*)?nights?", lowered)
        if m:
            return max(1, min(14, int(m.group(1))))
        for word, n in cls._WORD_TO_INT.items():
            if re.search(rf"\b{word}\b\s*(more\s*)?nights?", lowered):
                return n
        return 1
