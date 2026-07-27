"""Cancellation-risk (churn) model — LightGBM on the Antonio et al 2019 dataset.

Why LightGBM: fast, native categorical support, honest calibration out of the
box, and — most importantly — it's what a working analytics team would reach
for on a tabular problem this shape. Report AUROC / PR-AUC / Brier honestly on
a chronological holdout. Feature importance is the interpretability lever.

The model outputs `p_cancel ∈ [0, 1]`. The Profile agent surfaces the number
and its top drivers — it does not reinterpret or round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .dataset import CHURN_CATEGORICAL, CHURN_FEATURES

_LGB_PARAMS: Final[dict[str, object]] = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


@dataclass(slots=True)
class ChurnReport:
    n_train: int
    n_test: int
    train_cancellation_rate: float
    test_cancellation_rate: float
    auroc: float
    average_precision: float
    brier: float
    cutoff_date: str
    best_iteration: int
    feature_importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_cancellation_rate": self.train_cancellation_rate,
            "test_cancellation_rate": self.test_cancellation_rate,
            "auroc": self.auroc,
            "average_precision": self.average_precision,
            "brier": self.brier,
            "cutoff_date": self.cutoff_date,
            "best_iteration": self.best_iteration,
            "feature_importance": self.feature_importance,
        }


@dataclass(slots=True)
class ChurnModel:
    booster: lgb.Booster
    report: ChurnReport
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X[self.feature_names].copy()
        for col in CHURN_CATEGORICAL:
            if col in aligned.columns:
                aligned[col] = aligned[col].astype("category")
        return self.booster.predict(aligned, num_iteration=self.booster.best_iteration)

    def top_drivers(self, k: int = 5) -> list[tuple[str, float]]:
        items = sorted(
            self.report.feature_importance.items(), key=lambda kv: kv[1], reverse=True
        )
        return items[:k]


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cutoff_date: pd.Timestamp,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 25,
) -> ChurnModel:
    train_ds = lgb.Dataset(
        X_train,
        label=y_train.to_numpy(),
        categorical_feature=list(CHURN_CATEGORICAL),
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_test,
        label=y_test.to_numpy(),
        categorical_feature=list(CHURN_CATEGORICAL),
        reference=train_ds,
        free_raw_data=False,
    )
    booster = lgb.train(
        _LGB_PARAMS,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_score = booster.predict(X_test, num_iteration=booster.best_iteration)
    auroc = float(roc_auc_score(y_test, y_score))
    ap = float(average_precision_score(y_test, y_score))
    brier = float(brier_score_loss(y_test, y_score))

    gain = booster.feature_importance(importance_type="gain")
    total_gain = float(gain.sum()) or 1.0
    importance = {
        name: float(g) / total_gain
        for name, g in zip(CHURN_FEATURES, gain, strict=True)
    }

    report = ChurnReport(
        n_train=len(y_train),
        n_test=len(y_test),
        train_cancellation_rate=float(y_train.mean()),
        test_cancellation_rate=float(y_test.mean()),
        auroc=auroc,
        average_precision=ap,
        brier=brier,
        cutoff_date=cutoff_date.strftime("%Y-%m-%d"),
        best_iteration=int(booster.best_iteration or booster.num_trees()),
        feature_importance=importance,
    )
    return ChurnModel(
        booster=booster, report=report, feature_names=list(CHURN_FEATURES)
    )
