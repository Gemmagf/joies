"""Customer-Value bander.

The Antonio dataset does not carry a stable guest id, so a true multi-booking
CLV is not recoverable. What we can compute honestly:

  single_booking_revenue = adr × total_nights
  repeat_multiplier      = 1 + previous_bookings_not_canceled × loyalty_weight
  value_score            = single_booking_revenue × repeat_multiplier

We then band value_score into quartiles fit on the training set. This gives the
Profile agent a Bronze/Silver/Gold/Platinum tier grounded in observed revenue.

We explicitly do NOT call this "CLV" in surfaced copy — README and the agent
call it "value band" and cite this file. Honesty over marketing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

BANDS: Final[tuple[str, ...]] = ("Bronze", "Silver", "Gold", "Platinum")
_LOYALTY_WEIGHT: Final[float] = 0.15


@dataclass(slots=True)
class CLVReport:
    n_fit: int
    quartile_cutoffs: list[float]
    band_distribution: dict[str, int] = field(default_factory=dict)
    mean_revenue_by_band: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_fit": self.n_fit,
            "quartile_cutoffs": self.quartile_cutoffs,
            "band_distribution": self.band_distribution,
            "mean_revenue_by_band": self.mean_revenue_by_band,
            "loyalty_weight": _LOYALTY_WEIGHT,
        }


@dataclass(slots=True)
class CLVBander:
    cutoffs: list[float]
    report: CLVReport

    def score(self, df: pd.DataFrame) -> pd.Series:
        rev = df["adr"] * (df["stays_in_weekend_nights"] + df["stays_in_week_nights"])
        mult = 1.0 + df["previous_bookings_not_canceled"] * _LOYALTY_WEIGHT
        return (rev * mult).astype(float)

    def band(self, df: pd.DataFrame) -> pd.Series:
        scores = self.score(df).to_numpy()
        edges = np.array([-np.inf, *self.cutoffs, np.inf])
        idx = np.clip(np.searchsorted(edges[1:-1], scores, side="right"), 0, 3)
        return pd.Series([BANDS[i] for i in idx], index=df.index, name="value_band")


def fit_clv_bander(df: pd.DataFrame) -> CLVBander:
    rev = df["adr"] * (df["stays_in_weekend_nights"] + df["stays_in_week_nights"])
    mult = 1.0 + df["previous_bookings_not_canceled"] * _LOYALTY_WEIGHT
    scores = (rev * mult).astype(float).to_numpy()

    q = np.quantile(scores, [0.25, 0.5, 0.75])
    cutoffs = [float(v) for v in q]

    bander = CLVBander(
        cutoffs=cutoffs,
        report=CLVReport(n_fit=len(scores), quartile_cutoffs=cutoffs),
    )
    bands = bander.band(df)
    bander.report.band_distribution = bands.value_counts().to_dict()
    bander.report.mean_revenue_by_band = {
        b: float(rev[bands == b].mean()) if (bands == b).any() else 0.0 for b in BANDS
    }
    return bander
