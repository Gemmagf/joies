"""End-to-end eval runner that combines retrieval + groundedness + calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..data_loader import load_catalog, load_heritage
from .calibration import CalibrationReport, calibration_curve
from .groundedness import GroundednessReport, check_groundedness
from .retrieval import RetrievalMetrics, evaluate_retrieval


@dataclass(slots=True)
class EvalReport:
    timestamp: datetime
    catalog_retrieval: RetrievalMetrics
    catalog_hybrid_retrieval: RetrievalMetrics
    heritage_retrieval: RetrievalMetrics
    groundedness_samples: list[GroundednessReport] = field(default_factory=list)
    calibration: CalibrationReport | None = None


def run_full_eval(*, include_groundedness_smoke: bool = True) -> EvalReport:
    """Run the deterministic parts of the eval — no Anthropic API required.

    Calibration is loaded from `data/eval/calibration_pairs.jsonl` if present;
    otherwise it's left as None and the dashboard shows a placeholder.
    """
    catalog_metrics = evaluate_retrieval(target="catalog")
    catalog_hybrid_metrics = evaluate_retrieval(target="catalog_hybrid")
    heritage_metrics = evaluate_retrieval(target="heritage")

    groundedness_samples: list[GroundednessReport] = []
    if include_groundedness_smoke:
        catalog = load_catalog()
        heritage = load_heritage()
        sample_piece = catalog[0]
        sample_heritage = heritage[0]
        valid_reply = (
            f"{sample_piece.name.en} — {sample_piece.id} (CHF {sample_piece.price_chf:,.0f}) "
            f"is the founding silhouette [{sample_heritage.id}]."
        )
        invalid_reply = (
            "The Mythical Alhambra in jade — VCA-FAKE-001 (CHF 99,999) is a recent edition "
            "[HER-FAKE-2099]."
        )
        groundedness_samples.append(
            check_groundedness(
                valid_reply,
                catalog_evidence=[sample_piece],
                heritage_evidence=[sample_heritage],
            )
        )
        groundedness_samples.append(check_groundedness(invalid_reply))

    calibration_path = get_settings().data_dir / "eval" / "calibration_pairs.jsonl"
    calibration_report: CalibrationReport | None = None
    if calibration_path.exists():
        from .calibration import load_calibration_pairs

        pairs = load_calibration_pairs(calibration_path)
        calibration_report = calibration_curve(pairs)

    return EvalReport(
        timestamp=datetime.now(UTC),
        catalog_retrieval=catalog_metrics,
        catalog_hybrid_retrieval=catalog_hybrid_metrics,
        heritage_retrieval=heritage_metrics,
        groundedness_samples=groundedness_samples,
        calibration=calibration_report,
    )


def write_eval_report(report: EvalReport, path: Path) -> None:
    """Serialise the eval report to JSON for the dashboard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": report.timestamp.isoformat(),
        "catalog_retrieval": {
            "n_queries": report.catalog_retrieval.n_queries,
            "recall_at_1": report.catalog_retrieval.recall_at_1,
            "recall_at_3": report.catalog_retrieval.recall_at_3,
            "recall_at_5": report.catalog_retrieval.recall_at_5,
            "mrr": report.catalog_retrieval.mrr,
            "per_intent": report.catalog_retrieval.per_intent,
        },
        "catalog_hybrid_retrieval": {
            "n_queries": report.catalog_hybrid_retrieval.n_queries,
            "recall_at_1": report.catalog_hybrid_retrieval.recall_at_1,
            "recall_at_3": report.catalog_hybrid_retrieval.recall_at_3,
            "recall_at_5": report.catalog_hybrid_retrieval.recall_at_5,
            "mrr": report.catalog_hybrid_retrieval.mrr,
            "per_intent": report.catalog_hybrid_retrieval.per_intent,
        },
        "heritage_retrieval": {
            "n_queries": report.heritage_retrieval.n_queries,
            "recall_at_1": report.heritage_retrieval.recall_at_1,
            "recall_at_3": report.heritage_retrieval.recall_at_3,
            "recall_at_5": report.heritage_retrieval.recall_at_5,
            "mrr": report.heritage_retrieval.mrr,
            "per_intent": report.heritage_retrieval.per_intent,
        },
        "groundedness_samples": [
            {
                "reply": s.reply,
                "is_grounded": s.is_grounded,
                "hallucinations": s.hallucination_count,
                "findings": [
                    {"kind": f.kind, "detail": f.detail, "span": f.span} for f in s.findings
                ],
            }
            for s in report.groundedness_samples
        ],
        "calibration": (
            {
                "ece": report.calibration.expected_calibration_error,
                "brier": report.calibration.brier_score,
                "n_total": report.calibration.n_total,
                "bins": [
                    {
                        "lower": b.lower,
                        "upper": b.upper,
                        "n": b.n,
                        "avg_confidence": b.avg_confidence,
                        "accuracy": b.accuracy,
                    }
                    for b in report.calibration.bins
                ],
            }
            if report.calibration is not None
            else None
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
