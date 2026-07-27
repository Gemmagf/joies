"""Persist and load the trained analytics artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb

from ..config import get_settings
from .churn import ChurnModel, ChurnReport
from .clv import CLVBander, CLVReport
from .segmentation import GuestSegmenter, SegmentationReport


@dataclass(slots=True, frozen=True)
class AnalyticsArtifacts:
    churn: ChurnModel
    clv: CLVBander
    segmenter: GuestSegmenter


def save_all_artifacts(
    *,
    churn: ChurnModel,
    clv: CLVBander,
    segmenter: GuestSegmenter,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> Path:
    """Write model files + a single analytics.json summary. Returns the metrics path."""
    settings = get_settings()
    m_dir = models_dir or settings.models_dir
    met_dir = metrics_dir or settings.metrics_dir
    m_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    churn.booster.save_model(str(m_dir / "churn_lgbm.txt"))
    joblib.dump(
        {"cutoffs": clv.cutoffs},
        m_dir / "clv_bander.joblib",
    )
    joblib.dump(
        {"pipeline": segmenter.pipeline, "labels": segmenter.labels},
        m_dir / "guest_segmenter.joblib",
    )

    summary = {
        "churn": churn.report.as_dict(),
        "clv": clv.report.as_dict(),
        "segmentation": segmenter.report.as_dict(),
    }
    metrics_path = met_dir / "analytics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_path


def load_all_artifacts(
    *,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> AnalyticsArtifacts | None:
    settings = get_settings()
    m_dir = models_dir or settings.models_dir
    met_dir = metrics_dir or settings.metrics_dir
    metrics_path = met_dir / "analytics.json"
    if not metrics_path.exists():
        return None
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))

    booster = lgb.Booster(model_file=str(m_dir / "churn_lgbm.txt"))
    churn_report = _churn_report_from_dict(summary["churn"])
    churn = ChurnModel(
        booster=booster,
        report=churn_report,
        feature_names=list(booster.feature_name()),
    )

    clv_payload = joblib.load(m_dir / "clv_bander.joblib")
    clv_report = _clv_report_from_dict(summary["clv"])
    clv = CLVBander(cutoffs=clv_payload["cutoffs"], report=clv_report)

    seg_payload = joblib.load(m_dir / "guest_segmenter.joblib")
    seg_report = _seg_report_from_dict(summary["segmentation"])
    segmenter = GuestSegmenter(
        pipeline=seg_payload["pipeline"],
        labels={int(k): v for k, v in seg_payload["labels"].items()},
        report=seg_report,
    )
    return AnalyticsArtifacts(churn=churn, clv=clv, segmenter=segmenter)


def _churn_report_from_dict(d: dict) -> ChurnReport:
    return ChurnReport(
        n_train=d["n_train"],
        n_test=d["n_test"],
        train_cancellation_rate=d["train_cancellation_rate"],
        test_cancellation_rate=d["test_cancellation_rate"],
        auroc=d["auroc"],
        average_precision=d["average_precision"],
        brier=d["brier"],
        cutoff_date=d["cutoff_date"],
        best_iteration=d["best_iteration"],
        feature_importance=d["feature_importance"],
    )


def _clv_report_from_dict(d: dict) -> CLVReport:
    return CLVReport(
        n_fit=d["n_fit"],
        quartile_cutoffs=d["quartile_cutoffs"],
        band_distribution=d.get("band_distribution", {}),
        mean_revenue_by_band=d.get("mean_revenue_by_band", {}),
    )


def _seg_report_from_dict(d: dict) -> SegmentationReport:
    return SegmentationReport(
        n_fit=d["n_fit"],
        k=d["k"],
        inertia=d["inertia"],
        centroids=d.get("centroids", {}),
        sizes=d.get("sizes", {}),
    )
