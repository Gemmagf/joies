"""Dataset loader / cleaner / splitter — the foundation everything else stands on."""

from __future__ import annotations

import pandas as pd

from maison_concierge.analytics import (
    HotelBookings,
    load_bookings,
    prepare_features,
    split_chronological,
)
from maison_concierge.analytics.dataset import (
    CHURN_CATEGORICAL,
    CHURN_FEATURES,
    SEGMENT_FEATURES,
)


def test_load_bookings_is_populated_and_clean(bookings_full: HotelBookings) -> None:
    df = bookings_full.df
    assert bookings_full.n_rows > 100_000
    assert df["children"].isna().sum() == 0
    assert df["country"].isna().sum() == 0
    assert df["arrival_date"].notna().all()
    assert (df["adr"] >= 0).all() and (df["adr"] <= 1000).all()
    assert (df["total_nights"] >= 0).all()
    assert 0.0 < bookings_full.cancellation_rate < 1.0


def test_derived_fields_present(bookings_full: HotelBookings) -> None:
    df = bookings_full.df
    for col in ("total_nights", "party_size", "is_family", "is_weekend_heavy",
                "country_bucket", "revenue", "arrival_month"):
        assert col in df.columns, f"missing derived column {col}"
    for feat in CHURN_FEATURES + SEGMENT_FEATURES:
        assert feat in df.columns or feat in ("is_family", "is_weekend_heavy"), feat


def test_country_bucket_is_bounded(bookings_full: HotelBookings) -> None:
    buckets = bookings_full.df["country_bucket"].unique()
    assert len(buckets) <= 16  # top-15 + OTHER
    assert "OTHER" in buckets


def test_prepare_features_shapes_and_dtypes(bookings_full: HotelBookings) -> None:
    X, y = prepare_features(bookings_full)
    assert list(X.columns) == list(CHURN_FEATURES)
    assert len(X) == len(y) == bookings_full.n_rows
    for col in CHURN_CATEGORICAL:
        assert isinstance(X[col].dtype, pd.CategoricalDtype), col


def test_split_chronological_respects_time(bookings_full: HotelBookings) -> None:
    X_train, y_train, X_test, y_test, cutoff = split_chronological(
        bookings_full, test_fraction=0.2
    )
    assert len(X_train) + len(X_test) == bookings_full.n_rows
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    train_dates = bookings_full.df.sort_values("arrival_date")["arrival_date"]
    assert train_dates.iloc[len(X_train) - 1] <= cutoff
    assert train_dates.iloc[len(X_train)] >= cutoff


def test_load_bookings_deterministic(bookings_full: HotelBookings) -> None:
    again = load_bookings()
    assert again.n_rows == bookings_full.n_rows
    pd.testing.assert_series_equal(
        again.df["is_canceled"].reset_index(drop=True),
        bookings_full.df["is_canceled"].reset_index(drop=True),
    )
