"""Profile & Analytics agent — calls the Phase 1 models, never guesses.

Given a booking (as a Pandas row/dict / persona), returns a `ProfileSnapshot`
with cancellation probability, value band, and stay-type segment plus the
top model drivers. The orchestrator surfaces the snapshot to the user; the
composer is not allowed to invent, round, or reinterpret the numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ..analytics import AnalyticsArtifacts, load_all_artifacts
from ..analytics.dataset import CHURN_FEATURES, SEGMENT_FEATURES

_CHURN_BANDS: tuple[tuple[float, str], ...] = (
    (0.20, "low"),
    (0.50, "medium"),
    (1.01, "high"),
)


@dataclass(slots=True)
class ProfileSnapshot:
    persona_id: str | None
    p_cancel: float
    churn_band: str
    value_band: str
    revenue_estimate: float
    segment: str
    segment_note: str
    top_drivers: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        drivers = ", ".join(f"{name} ({share:.0%})" for name, share in self.top_drivers[:3])
        return (
            f"Cancellation probability: {self.p_cancel:.0%} ({self.churn_band}). "
            f"Value band: {self.value_band} (~€{self.revenue_estimate:.0f} this stay). "
            f"Segment: {self.segment} — {self.segment_note}. "
            f"Top model drivers: {drivers}."
        )


def _band_for(p: float) -> str:
    for threshold, name in _CHURN_BANDS:
        if p < threshold:
            return name
    return _CHURN_BANDS[-1][1]


def _explain_notes(row: pd.Series, top_drivers: list[tuple[str, float]]) -> list[str]:
    notes: list[str] = []
    if str(row.get("deposit_type", "")) == "Non Refund":
        notes.append(
            "The booking is on a non-refundable deposit — historically these convert "
            "at a much higher rate than flexible bookings."
        )
    if float(row.get("lead_time", 0)) > 180:
        notes.append(
            "Very long lead time (>180 days) — the model treats this as a cancellation-risk signal."
        )
    if int(row.get("previous_cancellations", 0)) > 0:
        notes.append(
            "This guest has cancelled before — the model weighs that heavily."
        )
    if int(row.get("total_of_special_requests", 0)) >= 3:
        notes.append(
            "The guest has already asked for several special requests — a positive engagement signal."
        )
    if not notes:
        top_names = [name for name, _ in top_drivers[:2]]
        notes.append(
            f"No strong booking-level signal; the risk is driven by {', '.join(top_names)}."
        )
    return notes


class ProfileAgent:
    """Wraps the trained analytics artifacts behind a per-booking API."""

    def __init__(self, artifacts: AnalyticsArtifacts | None = None) -> None:
        loaded = artifacts or load_all_artifacts()
        if loaded is None:
            raise RuntimeError(
                "Analytics artifacts not found. Run `python scripts/train_analytics.py` first."
            )
        self._artifacts = loaded

    @property
    def artifacts(self) -> AnalyticsArtifacts:
        return self._artifacts

    def snapshot(self, booking: pd.DataFrame | pd.Series | dict, *,
                 persona_id: str | None = None) -> ProfileSnapshot:
        row_df = self._as_frame(booking)
        churn = self._artifacts.churn
        clv = self._artifacts.clv
        segmenter = self._artifacts.segmenter

        p_cancel = float(churn.predict_proba(row_df[list(CHURN_FEATURES)])[0])
        band = _band_for(p_cancel)

        value_band = str(clv.band(row_df).iloc[0])
        revenue = float(clv.score(row_df).iloc[0])

        segment_labels = segmenter.assign(row_df[list(SEGMENT_FEATURES)])
        segment = str(segment_labels.iloc[0])
        # The segmenter labels a *cluster*; we tell the user which centroid features drove it.
        cluster_id = int(segmenter.pipeline.predict(
            row_df[list(SEGMENT_FEATURES)]
            .assign(is_family=row_df["is_family"].astype(int),
                    is_weekend_heavy=row_df["is_weekend_heavy"].astype(int))
            .to_numpy(dtype=float)
        )[0])
        segment_note = segmenter.describe(cluster_id)

        top_drivers = churn.top_drivers(5)
        notes = _explain_notes(row_df.iloc[0], top_drivers)

        return ProfileSnapshot(
            persona_id=persona_id,
            p_cancel=p_cancel,
            churn_band=band,
            value_band=value_band,
            revenue_estimate=revenue,
            segment=segment,
            segment_note=segment_note,
            top_drivers=top_drivers,
            notes=notes,
        )

    @staticmethod
    def _as_frame(booking: pd.DataFrame | pd.Series | dict) -> pd.DataFrame:
        if isinstance(booking, pd.DataFrame):
            df = booking.head(1).copy()
        elif isinstance(booking, pd.Series):
            df = booking.to_frame().T.copy()
        else:
            df = pd.DataFrame([booking])
        # Ensure categorical dtype for the LightGBM booster.
        from ..analytics.dataset import CHURN_CATEGORICAL
        for col in CHURN_CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df.reset_index(drop=True)
