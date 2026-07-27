"""Shared fixtures for analytics tests.

The full 118k-row load is cheap (~1s) but we cache it and sample per-test so
suites stay fast and independent.
"""

from __future__ import annotations

import pytest

from maison_concierge.analytics import HotelBookings, load_bookings


@pytest.fixture(scope="session")
def bookings_full() -> HotelBookings:
    return load_bookings()


@pytest.fixture(scope="module")
def bookings_sample(bookings_full: HotelBookings) -> HotelBookings:
    """~3,000-row deterministic sample, useful when a test needs the full pipeline
    to run fast (churn training, segmenter fit). Module-scoped so a downstream
    module-scoped fixture (e.g. `trained_on_sample`) can consume it."""
    sample_df = bookings_full.df.sample(n=3_000, random_state=42).reset_index(drop=True)
    return HotelBookings(df=sample_df)
