"""Reproducible training pipeline for the dataset-gate Parquet outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_NAMES = [
    "move_count",
    "mean_cpl",
    "top1_match_rate",
    "top3_match_rate",
    "mean_move_time_ms",
    "rated",
]
TARGET_COLUMN = "dataset_class"
GROUP_COLUMNS = ("username", "dataset_class")
MODEL_SEED = 2025


@dataclass(frozen=True)
class TrainingResult:
    """Serializable summary of one trained classifier."""

    model_name: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    calibration_curve: list[dict[str, float]]


def _dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import pandas as pd

    frames = [parquet.read_table(path).to_pandas() for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required_columns = [TARGET_COLUMN, *FEATURE_NAMES, *GROUP_COLUMNS]
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"training input is missing required columns: {', '.join(missing)}")
    labels = (frame[TARGET_COLUMN] == "engine_like").astype(int).to_numpy()
    features = frame[FEATURE_NAMES].copy()
    for column in FEATURE_NAMES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    groups = (
        frame["username"].fillna("<anonymous>").astype(str)
        + "::"
        + frame["dataset_class"].astype(str)
    ).to_numpy()
    return features.to_numpy(dtype=np.float64), labels, groups, len(frame)


def _split(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, ...]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=MODEL_SEED)
    train_valid, test = next(outer.split(features, labels, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=MODEL_SEED)
    relative_train, relative_valid = next(
        inner.split(features[train_valid], labels[train_valid], groups[train_valid])
    )
    train = train_valid[relative_train]
    valid = train_valid[relative_valid]
    return train, valid, test


def _thresholds(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [
        (
            precision_score(labels, probabilities >= threshold, zero_division=0)
            * recall_score(labels, probabilities >= threshold, zero_division=0),
            threshold,
        )
        for threshold in candidates
    ]
    suspicious = max(scores)[1]
    high_risk = max(
        (threshold for threshold in candidates if threshold > suspicious),
        default=min(0.95, suspicious + 0.05),
    )
    return {"suspicious": float(suspicious), "high_risk": float(high_risk)}


def _metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    fpr_limit = 0.01
    thresholds = np.linspace(0.0, 1.0, 1001)
    low_fpr_recall = max(
        (
            recall_score(labels, probabilities >= threshold, zero_division=0)
            for threshold in thresholds
            if np.mean((probabilities >= threshold) & (labels == 0)) / max(np.sum(labels == 0), 1)
            <= fpr_limit
        ),
        default=0.0,
    )
    curve_true, curve_pred = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
    curve = [
        {"predicted": float(predicted), "observed": float(observed)}
        for predicted, observed in zip(curve_pred, curve_true)
    ]
    return (
        {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "recall_at_1pct_fpr": float(low_fpr_recall),
            "precision_high_risk": float(
                precision_score(labels, probabilities >= 0.8, zero_division=0)
            ),
            "brier_score": float(brier_score_loss(labels, probabilities)),
        },
        curve,
    )


def train_dataset(input_dir: Path, report_path: Path) -> dict[str, Any]:
    """Train calibrated baseline and XGBoost models and write a JSON report."""
    paths = sorted(input_dir.glob("*_features.parquet"))
    if len(paths) < 2:
        raise ValueError("training requires both human and engine-like Parquet files")
    features, labels, groups, row_count = _load(paths)
    train, valid, test = _split(features, labels, groups)
    estimators: dict[str, Any] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=MODEL_SEED),
        ),
        "xgboost": XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=MODEL_SEED,
            n_jobs=1,
        ),
    }
    results: list[TrainingResult] = []
    for name, estimator in estimators.items():
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
        calibrated.fit(features[train], labels[train])
        validation_probabilities = calibrated.predict_proba(features[valid])[:, 1]
        test_probabilities = calibrated.predict_proba(features[test])[:, 1]
        metrics, curve = _metrics(test_probabilities, labels[test])
        results.append(
            TrainingResult(
                model_name=name,
                metrics=metrics,
                thresholds=_thresholds(validation_probabilities, labels[valid]),
                calibration_curve=curve,
            )
        )
    payload = {
        "report_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_hash": _dataset_hash(paths),
        "row_count": row_count,
        "feature_names": FEATURE_NAMES,
        "grouping": "username + dataset_class; no group crosses splits",
        "split_counts": {"train": len(train), "validation": len(valid), "test": len(test)},
        "models": [result.__dict__ for result in results],
        "random_seed": MODEL_SEED,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
