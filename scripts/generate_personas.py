"""Generate synthetic hospitality personas from the real bookings distribution.

Picks 20 stratified rows (both properties × canceled/kept × varied lead time)
and attaches an invented identity (name, email, booking reference). The
booking features themselves — lead time, deposit, ADR, party size — are the
dataset's real values. This is what the brief calls "synthetic personas from
real distributions".

Writes to `data/personas/hospitality_personas.json`.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from maison_concierge.analytics import load_bookings  # noqa: E402
from maison_concierge.analytics.dataset import CHURN_FEATURES, SEGMENT_FEATURES  # noqa: E402
from maison_concierge.hospitality.personas import Persona, save_personas  # noqa: E402

_PERSONA_POOL: list[tuple[str, str, str]] = [
    # (display_name, email, nationality_iso3)
    ("Sofia Marchetti",        "sofia.marchetti@example.example",   "ITA"),
    ("Alexandre Rousseau",     "a.rousseau@example.example",        "FRA"),
    ("Isabel Ferreira",        "isabel.ferreira@example.example",   "PRT"),
    ("James Whitcombe",        "james.whitcombe@example.example",   "GBR"),
    ("Kirsten Nyholm",         "kirsten.nyholm@example.example",    "DNK"),
    ("David Levine",           "d.levine@example.example",          "USA"),
    ("Marta Nowak",            "m.nowak@example.example",           "POL"),
    ("Anouk van der Berg",     "anouk.vdb@example.example",         "NLD"),
    ("Rafael Barros",          "r.barros@example.example",          "BRA"),
    ("Yuki Nakamura",          "yuki.nakamura@example.example",     "JPN"),
    ("Christine Bauer",        "c.bauer@example.example",           "DEU"),
    ("Elena Ruiz",             "elena.ruiz@example.example",        "ESP"),
    ("Malik Traoré",           "m.traore@example.example",          "FRA"),
    ("Priya Iyer",             "priya.iyer@example.example",        "IND"),
    ("Tomás Correia",          "t.correia@example.example",         "PRT"),
    ("Nikola Petrović",        "n.petrovic@example.example",        "SRB"),
    ("Aisha Farouk",           "a.farouk@example.example",          "EGY"),
    ("Béatrice Lefevre",       "b.lefevre@example.example",         "FRA"),
    ("Henry Osborne",          "h.osborne@example.example",         "GBR"),
    ("Solange Almeida",        "s.almeida@example.example",         "PRT"),
]

_PROPERTY_MAP = {"City Hotel": "Maison Lisboa", "Resort Hotel": "Maison Algarve"}


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    """Balanced across hotel × is_canceled × lead_time quartile.

    We sample indices per stratum (not rows), then reindex the full frame — this
    keeps the grouping columns (hotel, is_canceled) available on the result.
    """
    df = df.reset_index(drop=True).copy()
    df["lead_bucket"] = pd.qcut(df["lead_time"], q=4, labels=False, duplicates="drop")
    strata = df.groupby(["hotel", "is_canceled", "lead_bucket"], observed=True)
    per_stratum = max(1, n // max(1, len(strata)))
    idx: list[int] = []
    for _, group in strata:
        k = min(per_stratum, len(group))
        idx.extend(group.sample(k, random_state=seed).index.tolist())
    picked = df.loc[idx].reset_index(drop=True)
    if len(picked) < n:
        rest = df.drop(idx, errors="ignore")
        picked = pd.concat(
            [picked, rest.sample(n - len(picked), random_state=seed)],
            ignore_index=True,
        )
    return picked.head(n).reset_index(drop=True)


def _booking_features(row: pd.Series) -> dict[str, object]:
    keep = set(CHURN_FEATURES) | set(SEGMENT_FEATURES) | {
        "adr", "stays_in_weekend_nights", "stays_in_week_nights",
        "previous_bookings_not_canceled", "deposit_type", "hotel",
    }
    return {k: _coerce(row[k]) for k in keep if k in row}


def _coerce(v: object) -> object:
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


def main() -> int:
    bookings = load_bookings()
    df = bookings.df
    picked = _stratified_sample(df, n=len(_PERSONA_POOL))

    personas: list[Persona] = []
    for i, (row, identity) in enumerate(
        zip(picked.itertuples(index=False), _PERSONA_POOL, strict=True), start=1
    ):
        arrival = pd.Timestamp(row.arrival_date).date()
        nights = int(row.total_nights) or 1
        departure = arrival + timedelta(days=nights)
        property_name = _PROPERTY_MAP.get(str(row.hotel), "Maison")
        booking_ref = f"LM-{arrival.year}-{i:04d}"
        p = Persona(
            id=f"PER-{i:04d}",
            display_name=identity[0],
            email=identity[1],
            nationality=str(row.country),
            property=property_name,
            booking_ref=booking_ref,
            arrival_date=arrival,
            departure_date=departure,
            lead_time_days=int(row.lead_time),
            party_size=int(row.party_size),
            is_repeated_guest=bool(row.is_repeated_guest),
            previous_cancellations=int(row.previous_cancellations),
            booking=_booking_features(pd.Series(row._asdict())),
        )
        personas.append(p)

    path = save_personas(personas)
    print(f"[personas] wrote {len(personas)} personas to {path}")

    print("\n[personas] preview:")
    for p in personas[:5]:
        print(f"  {p.id}  {p.display_name:22s} {p.property:16s}  {p.stay_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _coerce_date(_v: date) -> str:  # kept for import stability if referenced
    return _v.isoformat()
