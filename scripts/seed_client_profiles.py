"""Generate 50 deterministic synthetic client profiles.

Usage:
    python scripts/seed_client_profiles.py [--n 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.config import get_settings  # noqa: E402
from maison_concierge.models import ClientProfile, ClientSegment, Collection  # noqa: E402

_FR_FIRSTS = ["Madeleine", "Camille", "Élodie", "Béatrice", "Antoine", "Henri", "Philippe", "Solange"]
_FR_LASTS = ["Dubois", "Berthier", "Garnier", "Marchetti", "Rey", "Laurent", "Vaillant", "Chastel"]
_EN_FIRSTS = ["Eleanor", "Margaret", "Catherine", "James", "Theodore", "Helena", "Adrian", "Isabel"]
_EN_LASTS = ["Worthington", "Halloway", "Pemberton", "Ashford", "Caldwell", "Whitcomb", "Hartford", "Sinclair"]

_CITIES = [
    ("Geneva", "fr"),
    ("Lausanne", "fr"),
    ("Zurich", "en"),
    ("Paris", "fr"),
    ("London", "en"),
    ("Monaco", "fr"),
    ("Milan", "en"),
    ("New York", "en"),
    ("Dubai", "en"),
    ("Hong Kong", "en"),
]

_SEGMENT_DISTRIBUTION = [
    (ClientSegment.PROSPECT, 0.36),
    (ClientSegment.ESTABLISHED, 0.40),
    (ClientSegment.VIP, 0.18),
    (ClientSegment.COLLECTOR, 0.06),
]


def _draw_segment(rng: random.Random) -> ClientSegment:
    r = rng.random()
    cumulative = 0.0
    for segment, p in _SEGMENT_DISTRIBUTION:
        cumulative += p
        if r < cumulative:
            return segment
    return _SEGMENT_DISTRIBUTION[-1][0]


def _profile_for_segment(rng: random.Random, segment: ClientSegment, idx: int) -> ClientProfile:
    use_fr = rng.random() < 0.55
    first = rng.choice(_FR_FIRSTS if use_fr else _EN_FIRSTS)
    last = rng.choice(_FR_LASTS if use_fr else _EN_LASTS)
    city, suggested_lang = rng.choice(_CITIES)
    locale = suggested_lang if rng.random() < 0.75 else ("fr" if suggested_lang == "en" else "en")

    if segment is ClientSegment.PROSPECT:
        tenure = round(rng.uniform(0.0, 1.0), 2)
        spend = round(rng.uniform(0.0, 5_000.0))
        collections = []
        if rng.random() < 0.4:
            collections = [rng.choice([Collection.ALHAMBRA, Collection.FRIVOLE])]
        last_appt_offset_days = rng.choice([None, rng.randint(30, 365)])
    elif segment is ClientSegment.ESTABLISHED:
        tenure = round(rng.uniform(1.5, 5.5), 1)
        spend = round(rng.uniform(5_000.0, 45_000.0))
        collections = rng.sample(list(Collection), k=rng.randint(1, 2))
        last_appt_offset_days = rng.randint(20, 240)
    elif segment is ClientSegment.VIP:
        tenure = round(rng.uniform(6.0, 14.0), 1)
        spend = round(rng.uniform(60_000.0, 400_000.0))
        collections = rng.sample(list(Collection), k=rng.randint(3, 4))
        last_appt_offset_days = rng.randint(10, 120)
    else:  # COLLECTOR
        tenure = round(rng.uniform(12.0, 30.0), 1)
        spend = round(rng.uniform(450_000.0, 2_500_000.0))
        collections = list(Collection)
        last_appt_offset_days = rng.randint(5, 60)

    last_at: datetime | None = None
    if last_appt_offset_days is not None:
        last_at = datetime.now(UTC) - timedelta(days=last_appt_offset_days)

    return ClientProfile(
        id=f"CL-{idx:04d}",
        display_name=f"{first} {last}",
        segment=segment,
        tenure_years=tenure,
        lifetime_spend_chf=spend,
        collections_owned=collections,
        preferred_language=locale,
        city=city,
        last_appointment_at=last_at,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic client profiles.")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    profiles = [
        _profile_for_segment(rng, _draw_segment(rng), i + 1) for i in range(args.n)
    ]

    settings = get_settings()
    path = settings.data_dir / "clients" / "profiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.model_dump(mode="json") for p in profiles], indent=2),
        encoding="utf-8",
    )

    seg_counts: dict[str, int] = {}
    for p in profiles:
        seg_counts[p.segment.value] = seg_counts.get(p.segment.value, 0) + 1
    print(f"Wrote {len(profiles)} profiles → {path}")
    print(f"  segments: {seg_counts}")
    avg_spend = sum(p.lifetime_spend_chf for p in profiles) / len(profiles)
    print(f"  avg lifetime spend: CHF {avg_spend:,.0f}")


if __name__ == "__main__":
    main()
