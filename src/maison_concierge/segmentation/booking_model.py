"""Booking probability model — lead scorer the SA team uses to prioritise follow-ups.

Trained on synthetic (conversation, client) pairs labelled with whether the
conversation produced a booking. Logistic regression on a small feature set: segment,
intent, log-spend, tenure, days since last appointment, conversation length, locale.

The model is small and interpretable on purpose. A senior DS reviewing the project
should be able to read the coefficients in five seconds and understand the policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure_years",
    "log_lifetime_spend_chf",
    "days_since_last_appointment",
    "n_turns",
    "intent_confidence",
]
CATEGORICAL_FEATURES = ["segment", "intent", "preferred_language"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(slots=True)
class BookingModelReport:
    n_train: int
    n_test: int
    accuracy: float
    roc_auc: float
    average_precision: float
    per_segment_precision: dict[str, float] = field(default_factory=dict)
    per_segment_recall: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report: str = ""


@dataclass(slots=True)
class BookingModel:
    pipeline: Pipeline
    report: BookingModelReport
    feature_names: list[str]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[ALL_FEATURES])[:, 1]

    def predict(self, df: pd.DataFrame, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=1_000, C=1.0, class_weight="balanced"),
            ),
        ]
    )


def _per_segment_metrics(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    precisions: dict[str, float] = {}
    recalls: dict[str, float] = {}
    for segment in sorted(df["segment"].unique()):
        mask = (df["segment"] == segment).to_numpy()
        if mask.sum() == 0:
            continue
        p, r, _, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], average="binary", zero_division=0.0
        )
        precisions[segment] = float(p)
        recalls[segment] = float(r)
    return precisions, recalls


def _feature_importance(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coefs = clf.coef_.ravel()
    return {name: float(coef) for name, coef in zip(feature_names, coefs, strict=True)}


def train_booking_model(
    df: pd.DataFrame, *, random_state: int = 42
) -> BookingModel:
    df = df.copy()
    df["log_lifetime_spend_chf"] = np.log1p(df["lifetime_spend_chf"])
    X = df[ALL_FEATURES]
    y = df["booked"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_score))
    avg_prec = float(average_precision_score(y_test, y_score))

    test_df = X_test.copy()
    test_df["segment"] = X_test["segment"].values
    per_seg_p, per_seg_r = _per_segment_metrics(test_df, y_test, y_pred)

    feature_names = list(
        pipeline.named_steps["prep"].get_feature_names_out(ALL_FEATURES)
    )
    importance = _feature_importance(pipeline, feature_names)

    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tp = int(((y_test == 1) & (y_pred == 1)).sum())

    report = BookingModelReport(
        n_train=len(y_train),
        n_test=len(y_test),
        accuracy=accuracy,
        roc_auc=roc_auc,
        average_precision=avg_prec,
        per_segment_precision=per_seg_p,
        per_segment_recall=per_seg_r,
        feature_importance=importance,
        confusion_matrix=[[tn, fp], [fn, tp]],
        classification_report=classification_report(
            y_test, y_pred, target_names=["no_booking", "booked"], zero_division=0.0
        ),
    )
    return BookingModel(pipeline=pipeline, report=report, feature_names=feature_names)


def save_booking_model(model: BookingModel, *, model_path: Path, report_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, model_path)
    payload: dict[str, Any] = {
        "n_train": model.report.n_train,
        "n_test": model.report.n_test,
        "accuracy": model.report.accuracy,
        "roc_auc": model.report.roc_auc,
        "average_precision": model.report.average_precision,
        "per_segment_precision": model.report.per_segment_precision,
        "per_segment_recall": model.report.per_segment_recall,
        "feature_importance": model.report.feature_importance,
        "confusion_matrix": model.report.confusion_matrix,
        "classification_report": model.report.classification_report,
        "feature_names": model.feature_names,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_booking_model(*, model_path: Path, report_path: Path) -> BookingModel | None:
    if not model_path.exists() or not report_path.exists():
        return None
    pipeline = joblib.load(model_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = BookingModelReport(
        n_train=payload["n_train"],
        n_test=payload["n_test"],
        accuracy=payload["accuracy"],
        roc_auc=payload["roc_auc"],
        average_precision=payload["average_precision"],
        per_segment_precision=payload["per_segment_precision"],
        per_segment_recall=payload["per_segment_recall"],
        feature_importance=payload["feature_importance"],
        confusion_matrix=payload["confusion_matrix"],
        classification_report=payload["classification_report"],
    )
    return BookingModel(
        pipeline=pipeline,
        report=report,
        feature_names=payload["feature_names"],
    )
