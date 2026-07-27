"""Stay-type segmentation via KMeans.

Features: total_nights, party_size, lead_time, adr, is_family, is_weekend_heavy.
Standardised, then KMeans with k=5 (a reasonable default for hospitality guest
mixes — solo business, couples, family holidays, planners, last-minute leisure).
Each cluster gets a human-readable label derived from its centroid.

The Profile agent calls `.assign(row)` for a single booking and gets back the
segment label plus a one-line "why" (the dominant feature that drove it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import SEGMENT_FEATURES

_K: int = 5
_RANDOM_STATE: int = 42


@dataclass(slots=True)
class SegmentationReport:
    n_fit: int
    k: int
    inertia: float
    centroids: dict[str, dict[str, float]] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_fit": self.n_fit,
            "k": self.k,
            "inertia": self.inertia,
            "centroids": self.centroids,
            "sizes": self.sizes,
        }


@dataclass(slots=True)
class GuestSegmenter:
    pipeline: Pipeline
    labels: dict[int, str]
    report: SegmentationReport

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        X = df[list(SEGMENT_FEATURES)].copy()
        X["is_family"] = X["is_family"].astype(int)
        X["is_weekend_heavy"] = X["is_weekend_heavy"].astype(int)
        return X.to_numpy(dtype=float)

    def assign(self, df: pd.DataFrame) -> pd.Series:
        clusters = self.pipeline.predict(self._features(df))
        return pd.Series(
            [self.labels[int(c)] for c in clusters], index=df.index, name="segment"
        )

    def describe(self, cluster_id: int) -> str:
        c = self.report.centroids[self.labels[cluster_id]]
        # Pick the two most distinctive features (largest absolute z-score vs mean).
        # Simple heuristic — enough for a one-line "why".
        top = sorted(c.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        return ", ".join(f"{k}={v:+.2f}sd" for k, v in top)


# Dominant-feature → label. Pick the biggest |z-score| in each centroid, then
# look up here. Sign matters: +is_weekend_heavy means "leans weekend", -lead_time
# means "books close to arrival". If the winner is muted (|z| < 0.35) we call it
# a standard stay — every real cluster in the fitted data has at least one
# feature past that threshold.
_LABEL_TABLE: Final[dict[tuple[str, str], str]] = {
    ("is_weekend_heavy", "+"): "weekend_getaway",
    ("is_weekend_heavy", "-"): "midweek_stay",
    ("lead_time", "+"): "early_planner",
    ("lead_time", "-"): "last_minute",
    ("is_family", "+"): "family_holiday",
    ("is_family", "-"): "adult_only",
    ("total_nights", "+"): "extended_stay",
    ("total_nights", "-"): "quick_stay",
    ("adr", "+"): "premium_stay",
    ("adr", "-"): "budget_stay",
    ("party_size", "+"): "group_travel",
    ("party_size", "-"): "solo_traveller",
}
_LABEL_MIN_Z: Final[float] = 0.35


def _label_cluster(centroid: dict[str, float]) -> str:
    ranked = sorted(centroid.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_feat, top_val = ranked[0]
    if abs(top_val) < _LABEL_MIN_Z:
        return "standard_stay"
    return _LABEL_TABLE.get((top_feat, "+" if top_val > 0 else "-"), "standard_stay")


def fit_guest_segmenter(df: pd.DataFrame) -> GuestSegmenter:
    X = df[list(SEGMENT_FEATURES)].copy()
    X["is_family"] = X["is_family"].astype(int)
    X["is_weekend_heavy"] = X["is_weekend_heavy"].astype(int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=_K, n_init=10, random_state=_RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(X.to_numpy(dtype=float))
    clusters = pipeline.predict(X.to_numpy(dtype=float))

    kmeans: KMeans = pipeline.named_steps["kmeans"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]

    # KMeans centres live in scaled space — that's already interpretable as
    # z-scores against the training mean/std, so we keep them as such.
    centroids_scaled = kmeans.cluster_centers_
    raw_centroids = [
        {feat: float(v) for feat, v in zip(SEGMENT_FEATURES, row, strict=True)}
        for row in centroids_scaled
    ]

    labels_seen: set[str] = set()
    labels: dict[int, str] = {}
    for cid, c in enumerate(raw_centroids):
        base = _label_cluster(c)
        label = base
        n = 2
        while label in labels_seen:
            label = f"{base}_{n}"
            n += 1
        labels_seen.add(label)
        labels[cid] = label

    # Report uses labels as keys so it's the same shape a caller reads elsewhere.
    _ = scaler  # kept referenced for clarity; the scaler is inside the pipeline.
    centroid_dict = {
        labels[i]: raw_centroids[i] for i in range(len(raw_centroids))
    }
    sizes = pd.Series(clusters).value_counts().to_dict()
    size_by_label = {labels[int(k)]: int(v) for k, v in sizes.items()}

    report = SegmentationReport(
        n_fit=len(clusters),
        k=_K,
        inertia=float(kmeans.inertia_),
        centroids=centroid_dict,
        sizes=size_by_label,
    )
    return GuestSegmenter(pipeline=pipeline, labels=labels, report=report)
