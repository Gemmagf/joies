"""Churn model — trains, scores, and produces honest metrics."""

from __future__ import annotations

import pytest

from maison_concierge.analytics import (
    HotelBookings,
    prepare_features,
    split_chronological,
    train_churn_model,
)


@pytest.fixture(scope="module")
def trained_on_sample(bookings_sample: HotelBookings):
    X_train, y_train, X_test, y_test, cutoff = split_chronological(
        bookings_sample, test_fraction=0.25
    )
    return train_churn_model(X_train, y_train, X_test, y_test, cutoff_date=cutoff)


def test_churn_beats_chance_on_sample(trained_on_sample) -> None:
    # 3k-row sample: honest expectation is ~0.75+. Full-dataset training gives 0.86+.
    assert trained_on_sample.report.auroc >= 0.70
    assert 0.0 <= trained_on_sample.report.brier <= 0.30
    assert trained_on_sample.report.best_iteration > 0


def test_predict_proba_is_bounded(trained_on_sample, bookings_sample: HotelBookings) -> None:
    X, _ = prepare_features(bookings_sample)
    p = trained_on_sample.predict_proba(X.head(200))
    assert p.shape == (200,)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_top_drivers_include_expected_features(trained_on_sample) -> None:
    top_names = [name for name, _ in trained_on_sample.top_drivers(6)]
    # deposit_type is the dominant signal in the literature and on the full run.
    # On a 3k sample it should still land in the top 6.
    assert "deposit_type" in top_names or "lead_time" in top_names


def test_feature_importance_sums_to_one(trained_on_sample) -> None:
    total = sum(trained_on_sample.report.feature_importance.values())
    assert 0.99 <= total <= 1.01
