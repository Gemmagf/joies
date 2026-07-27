"""Load, clean, and feature-engineer the Antonio et al 2019 hotel-bookings dataset.

Design rules:
- Cleaning is deterministic and pure — same CSV in, same DataFrame out.
- Feature engineering is separate from cleaning. A caller can inspect the
  cleaned frame before deciding what to model.
- The chronological split is the honest one for production: train on the past,
  score the future. A stratified random split is also exposed for tests /
  quick sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ..config import get_settings

RAW_COLUMNS_KEPT: Final[tuple[str, ...]] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
)

CHURN_NUMERIC: Final[tuple[str, ...]] = (
    "lead_time",
    "adr",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "days_in_waiting_list",
    "total_nights",
    "party_size",
    "is_repeated_guest",
)
CHURN_CATEGORICAL: Final[tuple[str, ...]] = (
    "hotel",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "country_bucket",
    "meal",
    "reserved_room_type",
    "arrival_month",
)
CHURN_FEATURES: Final[tuple[str, ...]] = CHURN_NUMERIC + CHURN_CATEGORICAL

SEGMENT_FEATURES: Final[tuple[str, ...]] = (
    "total_nights",
    "party_size",
    "lead_time",
    "adr",
    "is_family",
    "is_weekend_heavy",
)

# Top-N countries stay as themselves; the long tail collapses into OTHER. We
# keep 15 — enough to preserve the strong signal from PRT/GBR/FRA/ESP without
# blowing up one-hot dimensionality.
_TOP_COUNTRIES_N: Final[int] = 15


@dataclass(slots=True, frozen=True)
class HotelBookings:
    """Cleaned bookings frame with derived analytics fields.

    `df` has all columns from :data:`RAW_COLUMNS_KEPT` plus:
    `arrival_date` (datetime), `arrival_month` (str), `total_nights` (int),
    `party_size` (int), `is_family` (bool), `is_weekend_heavy` (bool),
    `country_bucket` (top-N or OTHER), `revenue` (float, ADR × nights).
    """

    df: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def cancellation_rate(self) -> float:
        return float(self.df["is_canceled"].mean())


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[list(RAW_COLUMNS_KEPT)].copy()

    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("UNK").astype(str)
    df["meal"] = df["meal"].replace({"Undefined": "SC"})

    df["adr"] = df["adr"].clip(lower=0.0, upper=1000.0)

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
        errors="coerce",
    )
    df["arrival_month"] = df["arrival_date_month"]

    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    ).astype(int)
    df["party_size"] = (df["adults"] + df["children"] + df["babies"]).astype(int)
    df["is_family"] = (df["children"] + df["babies"]) > 0
    df["is_weekend_heavy"] = df["stays_in_weekend_nights"] > df["stays_in_week_nights"]

    df["revenue"] = (df["adr"] * df["total_nights"]).astype(float)

    return df


def _bucket_country(df: pd.DataFrame) -> pd.DataFrame:
    top = df["country"].value_counts().nlargest(_TOP_COUNTRIES_N).index
    df["country_bucket"] = np.where(df["country"].isin(top), df["country"], "OTHER")
    return df


def load_bookings(path: Path | None = None) -> HotelBookings:
    """Load, clean, and derive analytics fields. Reads .csv or .csv.gz."""
    src = path or get_settings().bookings_path
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(
            f"Bookings dataset not found at {src}. "
            "Run `python scripts/prepare_data.py` or set BOOKINGS_PATH."
        )
    df = pd.read_csv(src)
    df = _clean(df)
    df = _bucket_country(df)
    df = df.dropna(subset=["arrival_date"]).reset_index(drop=True)
    return HotelBookings(df=df)


def prepare_features(bookings: HotelBookings) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the churn feature matrix and target from a cleaned bookings frame."""
    df = bookings.df
    X = df[list(CHURN_FEATURES)].copy()
    for col in CHURN_CATEGORICAL:
        X[col] = X[col].astype("category")
    y = df["is_canceled"].astype(int)
    return X, y


def split_chronological(
    bookings: HotelBookings,
    *,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    """Chronological train/test split. Returns (X_train, y_train, X_test, y_test, cutoff).

    Sorted by arrival_date, then the last `test_fraction` becomes the test set.
    This mirrors production: fit on the past, predict the future.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    df = bookings.df.sort_values("arrival_date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff = df["arrival_date"].iloc[cutoff_idx]

    train = HotelBookings(df=df.iloc[:cutoff_idx].reset_index(drop=True))
    test = HotelBookings(df=df.iloc[cutoff_idx:].reset_index(drop=True))
    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)
    return X_train, y_train, X_test, y_test, cutoff
