"""Confidence calibration analysis.

Given (predicted_confidence, was_correct) pairs from intent classification, bin into
deciles and compute the empirical accuracy per bin. A well-calibrated classifier has
accuracy ≈ confidence; the deviation is the calibration gap. Plotting these together
produces the reliability diagram.

Pairs are sourced from the eval runner (`runner.py`) or from a stored JSONL of past
predictions; the dashboard reads the latter to render without re-running the eval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np


@dataclass(slots=True)
class CalibrationBin:
    lower: float
    upper: float
    n: int
    avg_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return self.accuracy - self.avg_confidence


@dataclass(slots=True)
class CalibrationReport:
    bins: list[CalibrationBin]
    expected_calibration_error: float
    brier_score: float
    n_total: int


def calibration_curve(
    pairs: list[tuple[float, bool]], *, n_bins: int = 10
) -> CalibrationReport:
    """Compute reliability-diagram bins, ECE and Brier score over confidence pairs."""
    if not pairs:
        return CalibrationReport(bins=[], expected_calibration_error=0.0, brier_score=0.0, n_total=0)

    confidences = np.array([p[0] for p in pairs], dtype=float)
    correct = np.array([1.0 if p[1] else 0.0 for p in pairs], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    ece = 0.0
    total = len(pairs)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = (
            (confidences >= lower) & (confidences < upper)
            if i < n_bins - 1
            else (confidences >= lower) & (confidences <= upper)
        )
        n = int(in_bin.sum())
        if n == 0:
            bins.append(CalibrationBin(lower=lower, upper=upper, n=0, avg_confidence=0.0, accuracy=0.0))
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        bins.append(CalibrationBin(lower=lower, upper=upper, n=n, avg_confidence=avg_conf, accuracy=acc))
        ece += (n / total) * abs(acc - avg_conf)

    brier = float(np.mean((confidences - correct) ** 2))
    return CalibrationReport(
        bins=bins,
        expected_calibration_error=ece,
        brier_score=brier,
        n_total=total,
    )


def load_calibration_pairs(path: Path) -> list[tuple[float, bool]]:
    """Read (confidence, correct) pairs from a JSONL file written by the eval runner."""
    if not path.exists():
        return []
    pairs: list[tuple[float, bool]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "confidence" in row and "correct" in row:
            pairs.append((float(row["confidence"]), bool(row["correct"])))
    return pairs


def _quick_summary(report: CalibrationReport) -> str:
    if report.n_total == 0:
        return "No calibration data."
    return (
        f"ECE={report.expected_calibration_error:.3f}  "
        f"Brier={report.brier_score:.3f}  N={report.n_total}  "
        f"avg_acc={mean(b.accuracy for b in report.bins if b.n):.3f}"
    )
