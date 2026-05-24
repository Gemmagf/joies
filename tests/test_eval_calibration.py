import random

from maison_concierge.eval.calibration import calibration_curve


def test_perfectly_calibrated_classifier_has_zero_ece():
    rng = random.Random(0)
    pairs: list[tuple[float, bool]] = []
    for _ in range(5_000):
        confidence = round(rng.random(), 3)
        correct = rng.random() < confidence
        pairs.append((confidence, correct))
    report = calibration_curve(pairs, n_bins=10)
    assert report.expected_calibration_error < 0.05
    assert report.n_total == 5_000


def test_overconfident_classifier_has_positive_ece():
    pairs = [(0.95, False)] * 100 + [(0.95, True)] * 20
    report = calibration_curve(pairs, n_bins=10)
    assert report.expected_calibration_error > 0.5


def test_empty_pairs_returns_empty_report():
    report = calibration_curve([])
    assert report.n_total == 0
    assert report.expected_calibration_error == 0.0
    assert report.bins == []


def test_brier_score_is_zero_for_perfect_predictions():
    pairs = [(1.0, True)] * 50 + [(0.0, False)] * 50
    report = calibration_curve(pairs)
    assert report.brier_score == 0.0
