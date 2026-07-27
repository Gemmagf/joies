"""CLV bander — quartile bands are monotone and cover every row."""

from __future__ import annotations

from maison_concierge.analytics import HotelBookings, fit_clv_bander
from maison_concierge.analytics.clv import BANDS


def test_bander_assigns_a_band_to_every_row(bookings_full: HotelBookings) -> None:
    bander = fit_clv_bander(bookings_full.df)
    bands = bander.band(bookings_full.df)
    assert len(bands) == bookings_full.n_rows
    assert set(bands.unique()).issubset(set(BANDS))


def test_bands_are_monotone_in_score(bookings_full: HotelBookings) -> None:
    bander = fit_clv_bander(bookings_full.df)
    scores = bander.score(bookings_full.df)
    bands = bander.band(bookings_full.df)

    means = {b: float(scores[bands == b].mean()) for b in BANDS if (bands == b).any()}
    ordered = [means[b] for b in BANDS if b in means]
    assert ordered == sorted(ordered), f"band means not monotone: {means}"


def test_cutoffs_are_strictly_increasing(bookings_full: HotelBookings) -> None:
    bander = fit_clv_bander(bookings_full.df)
    assert bander.cutoffs == sorted(bander.cutoffs)
    assert bander.cutoffs[0] < bander.cutoffs[1] < bander.cutoffs[2]


def test_platinum_bookings_exist(bookings_full: HotelBookings) -> None:
    bander = fit_clv_bander(bookings_full.df)
    bands = bander.band(bookings_full.df)
    # ~25% should be Platinum by construction.
    frac = (bands == "Platinum").mean()
    assert 0.20 <= frac <= 0.30
