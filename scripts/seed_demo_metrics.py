"""Generate a realistic synthetic event log so the stakeholder dashboard has content.

The dashboard reads events from `metrics_dir/events.jsonl`. This script writes ~50
events distributed across intents, escalations, tool calls, appointments, leads, and
ratings to mimic two weeks of light traffic.

Usage:
    python scripts/seed_demo_metrics.py            # append to existing log
    python scripts/seed_demo_metrics.py --reset    # truncate and rewrite
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.models import ClientIntent, EscalationReason  # noqa: E402
from maison_concierge.observability import Event, get_recorder  # noqa: E402

INTENT_DISTRIBUTION: list[tuple[ClientIntent, int]] = [
    (ClientIntent.GIFT, 8),
    (ClientIntent.BROWSE, 5),
    (ClientIntent.HERITAGE_INQUIRY, 5),
    (ClientIntent.CELEBRATION, 4),
    (ClientIntent.APPOINTMENT, 3),
    (ClientIntent.PRICE_INQUIRY, 2),
    (ClientIntent.INVESTMENT_PIECE, 2),
    (ClientIntent.UNKNOWN, 1),
]

ESCALATION_DISTRIBUTION: list[tuple[EscalationReason, int]] = [
    (EscalationReason.LOW_CONFIDENCE, 4),
    (EscalationReason.EXPLICIT_REQUEST, 2),
    (EscalationReason.OUT_OF_SCOPE, 1),
]

LEAD_TIERS = [("hot", 3), ("warm", 5), ("watch", 10)]
RATINGS = [5, 5, 5, 5, 4, 4, 4, 3]


def _backdated(days_ago_max: int = 14) -> datetime:
    delta = timedelta(
        days=random.uniform(0, days_ago_max),
        hours=random.uniform(0, 24),
    )
    return datetime.now(UTC) - delta


def _conversation_ids(count: int) -> list[str]:
    return [str(uuid.uuid4()) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the dashboard with synthetic events.")
    parser.add_argument("--reset", action="store_true", help="Truncate the event log first.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    random.seed(args.seed)

    recorder = get_recorder()
    if args.reset and recorder.path.exists():
        recorder.path.unlink()
        recorder.path.parent.mkdir(parents=True, exist_ok=True)

    total_conversations = sum(count for _, count in INTENT_DISTRIBUTION)
    ids = _conversation_ids(total_conversations)
    cursor = 0

    for intent, count in INTENT_DISTRIBUTION:
        for _ in range(count):
            cid = ids[cursor]
            cursor += 1
            ts = _backdated()
            recorder.record(Event(type="conversation_started", conversation_id=cid, timestamp=ts))
            recorder.record(
                Event(
                    type="intent_detected",
                    conversation_id=cid,
                    timestamp=ts + timedelta(seconds=2),
                    payload={
                        "intent": intent.value,
                        "confidence": round(random.uniform(0.72, 0.96), 2),
                    },
                )
            )
            if intent in (ClientIntent.GIFT, ClientIntent.CELEBRATION, ClientIntent.BROWSE):
                recorder.record(
                    Event(
                        type="tool_called",
                        conversation_id=cid,
                        timestamp=ts + timedelta(seconds=4),
                        payload={"tool": "catalog_rag", "ok": True},
                    )
                )
            if intent is ClientIntent.HERITAGE_INQUIRY:
                recorder.record(
                    Event(
                        type="tool_called",
                        conversation_id=cid,
                        timestamp=ts + timedelta(seconds=4),
                        payload={"tool": "heritage_rag", "ok": True},
                    )
                )

    for reason, count in ESCALATION_DISTRIBUTION:
        for _ in range(count):
            cid = random.choice(ids)
            recorder.record(
                Event(
                    type="escalation",
                    conversation_id=cid,
                    timestamp=_backdated(),
                    payload={"reason": reason.value},
                )
            )

    for _ in range(5):
        cid = random.choice(ids)
        recorder.record(
            Event(
                type="appointment_booked",
                conversation_id=cid,
                timestamp=_backdated(),
                payload={
                    "reference": f"VCA-APT-{uuid.uuid4().hex[:8].upper()}",
                    "boutique": random.choice(["Geneva — Rue du Rhône", "Paris — Place Vendôme"]),
                    "channel": random.choice(["boutique", "video_call", "private_residence"]),
                },
            )
        )

    for tier, count in LEAD_TIERS:
        for _ in range(count):
            cid = random.choice(ids)
            recorder.record(
                Event(
                    type="lead_flagged",
                    conversation_id=cid,
                    timestamp=_backdated(),
                    payload={"tier": tier},
                )
            )

    for rating in random.sample(RATINGS, k=min(6, len(RATINGS))):
        cid = random.choice(ids)
        recorder.record(
            Event(
                type="rating_submitted",
                conversation_id=cid,
                timestamp=_backdated(),
                payload={"rating": rating},
            )
        )

    snapshot = recorder.snapshot()
    print(f"Wrote events to {recorder.path}")
    print(f"  conversations:       {snapshot.total_conversations}")
    print(f"  escalation rate:     {snapshot.escalation_rate:.0%}")
    print(f"  appointments:        {snapshot.appointments_booked}")
    print(f"  avg rating:          {snapshot.avg_rating:.2f}" if snapshot.avg_rating else "  avg rating:          n/a")
    print(f"  top intent:          {max(snapshot.intents, key=snapshot.intents.get)}")
    print(f"  leads by tier:       {snapshot.leads_by_tier}")


if __name__ == "__main__":
    main()
