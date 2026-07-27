"""Guest segmenter — labels every row, produces distinct clusters."""

from __future__ import annotations

from maison_concierge.analytics import HotelBookings, fit_guest_segmenter


def test_segmenter_assigns_a_label_to_every_row(bookings_full: HotelBookings) -> None:
    segmenter = fit_guest_segmenter(bookings_full.df)
    assigned = segmenter.assign(bookings_full.df)
    assert len(assigned) == bookings_full.n_rows
    assert assigned.isna().sum() == 0


def test_segmenter_produces_five_distinct_clusters(bookings_full: HotelBookings) -> None:
    segmenter = fit_guest_segmenter(bookings_full.df)
    assert len(segmenter.report.sizes) == 5
    # No cluster may be trivially empty; no single cluster may hog >70%.
    for name, size in segmenter.report.sizes.items():
        frac = size / bookings_full.n_rows
        assert 0.02 <= frac <= 0.70, f"cluster {name} degenerate at {frac:.2%}"


def test_labels_are_from_the_known_taxonomy(bookings_full: HotelBookings) -> None:
    segmenter = fit_guest_segmenter(bookings_full.df)
    known_prefixes = {
        "weekend_getaway", "midweek_stay", "early_planner", "last_minute",
        "family_holiday", "adult_only", "extended_stay", "quick_stay",
        "premium_stay", "budget_stay", "group_travel", "solo_traveller",
        "standard_stay",
    }
    for label in segmenter.report.sizes:
        base = label.rsplit("_", 1)[0] if label[-1].isdigit() else label
        assert base in known_prefixes or label in known_prefixes, label
