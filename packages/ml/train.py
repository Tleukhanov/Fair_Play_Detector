"""Train, calibrate, and report the ChessFraud game-level risk model.

The serving artifact is a **game-level** model: one row per player-game,
aggregated from the same ChessFraud decision points that the per-move feature
contract uses.  The protocol is deliberately conservative:

* the split is 60/20/20 and **player-disjoint** (``GroupShuffleSplit``);
* calibration is fitted on the validation split with a **frozen** base
  estimator, so no validation row ever trains the base model and no test row is
  used for calibration or threshold selection;
* operationally relevant thresholds are chosen on validation at target false
  positive rates of 1% and 5%, never on test;
* uncertainty is bootstrapped **by player**, not by row;
* a fail-closed leakage gate rejects forbidden columns before any fitting.

Two baselines are reported: a univariate heuristic (``top_1_match_rate`` and
inverse mean centipawn loss) and :class:`packages.ml.baseline.BaselineScorer`.
The report states explicitly whether the trained model beats them, including a
player-bootstrap confidence interval for the AUC difference.

A secondary, research-only per-move model (the 12-feature contract from
:mod:`packages.ml.feature_contract`) is trained with ``GroupKFold`` by player.
It is **not** part of the serving artifact.

Usage::

    python -m packages.ml.train \
        --data-dir /path/to/chessfraud \
        --output-dir data/models/chessfraud-v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # type: ignore[import-untyped]
from sklearn.frozen import FrozenEstimator  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from packages.ml.data import (
    CHESSFRAUD_FILENAME,
    load_chessfraud,
    resolve_data_path,
)
from packages.ml.feature_contract import (
    FEATURE_NAMES,
    MIN_HALF_MOVE,
    FeatureContractError,
    validate_feature_columns,
    validate_player_disjoint,
)
from packages.ml.game_features import (
    GAME_FEATURE_NAMES,
    GAME_FEATURE_SCHEMA_VERSION,
    build_game_frame,
    game_dataset_from_frame,
    game_from_record,
    validate_game_feature_names,
)

# numpy 2.x stubs use ``type`` statements that mypy refuses to parse while the
# project targets Python 3.11, and xgboost's typed modules import numpy.  A static
# import therefore aborts the whole mypy run before it can check this module.
# Resolving both dynamically keeps the runtime behavior identical and lets the
# strict mypy gate verify the annotations defined here.
np: Any = importlib.import_module("numpy")
_XGBOOST: Any = importlib.import_module("xgboost")
XGBClassifier: Any = _XGBOOST.XGBClassifier

MODEL_VERSION = "chessfraud-v1"
MODEL_SEED = 2025
SERVING_MODEL = "xgboost"
CALIBRATION_METHOD = "sigmoid"
BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_SEED = 2025
ENGINE_NAME = "stockfish"
ENGINE_VERSION = "17"
ANALYSIS_DEPTH = 15

#: Elo bins used only to *stratify the evaluation*; Elo is never a model feature.
ELO_BIN_EDGES: tuple[int, ...] = (1000, 1400, 1600, 1800, 2000, 2200, 2400, 2600)

PREVALENCE_CAVEAT = (
    "ChessFraud is a controlled tournament cohort with a player-game cheating "
    "prevalence around 40%. Live traffic has prevalence well below 1%, so "
    "precision values here are NOT operational precision; always re-evaluate "
    "precision at the deployment prevalence. Recall at a fixed FPR is the "
    "comparable quantity."
)

def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_leakage(
    columns: Iterable[str],
    *,
    expected: Sequence[str] = GAME_FEATURE_NAMES,
) -> None:
    """Fail closed unless ``columns`` are exactly the game feature contract.

    Checked before fitting.  Forbidden columns (labels, identity, raw Elo) and
    unknown/extra columns both raise :class:`FeatureContractError`.
    """

    actual = tuple(columns)
    validate_game_feature_names(actual)
    if actual != tuple(expected):
        missing = [name for name in expected if name not in actual]
        unexpected = [name for name in actual if name not in expected]
        raise FeatureContractError(
            f"game feature columns do not match the training contract; "
            f"missing={missing}, unexpected={unexpected}"
        )


def split_by_player(
    groups: Sequence[object],
    *,
    seed: int = MODEL_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices with disjoint players (60/20/20)."""

    group_array = np.asarray(list(groups), dtype=object)
    if len(np.unique(group_array)) < 3:
        raise ValueError("player-disjoint splitting requires at least three players")
    dummy = np.zeros((len(group_array), 1), dtype=float)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_valid_index, test_index = next(
        outer.split(dummy, np.zeros(len(group_array)), group_array)
    )
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    relative_train, relative_valid = next(
        inner.split(
            dummy[train_valid_index],
            np.zeros(len(train_valid_index)),
            group_array[train_valid_index],
        )
    )
    train_index = train_valid_index[relative_train]
    valid_index = train_valid_index[relative_valid]
    _check_disjoint(group_array, train_index, valid_index, test_index)
    return train_index, valid_index, test_index


def _check_disjoint(
    groups: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    validate_player_disjoint(
        groups[train_index], groups[valid_index], split_name="train/validation"
    )
    validate_player_disjoint(groups[train_index], groups[test_index], split_name="train/test")
    validate_player_disjoint(
        groups[valid_index], groups[test_index], split_name="validation/test"
    )


def _estimators(seed: int) -> dict[str, Any]:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
    }


def _fit_calibrated(
    estimator: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
) -> Any:
    """Fit the base model on train, then a sigmoid calibrator on validation.

    ``FrozenEstimator`` guarantees that calibration does not refit the base model
    on validation rows; validation players are disjoint from train players.
    """

    if len(np.unique(train_labels)) < 2:
        raise ValueError("training split contains a single class")
    if len(np.unique(valid_labels)) < 2:
        raise ValueError("validation split contains a single class")
    estimator.fit(train_features, train_labels)
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=CALIBRATION_METHOD)
    calibrated.fit(valid_features, valid_labels)
    return calibrated


def predict_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities as a float array."""

    probabilities = model.predict_proba(features)
    return np.asarray(probabilities, dtype=float)[:, 1]


def _base_estimator(calibrated: Any) -> Any:
    wrapper = getattr(calibrated, "estimator", None)
    if wrapper is not None and hasattr(wrapper, "estimator"):
        return wrapper.estimator
    return wrapper


def feature_importance(model: Any, names: Sequence[str]) -> list[dict[str, float | str]]:
    """Global importance for XGBoost (gain-free split importance) or logreg."""

    base = _base_estimator(model)
    if base is None:
        return []
    if hasattr(base, "feature_importances_"):
        values = np.asarray(base.feature_importances_, dtype=float)
    elif hasattr(base, "named_steps"):
        coefficients = np.asarray(
            base.named_steps["logisticregression"].coef_[0], dtype=float
        )
        values = np.abs(coefficients)
    else:
        return []
    total = float(values.sum())
    normalized = values / total if total > 0 else values
    pairs = sorted(
        zip(names, (float(value) for value in normalized)),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"name": name, "importance": importance} for name, importance in pairs]


def _core_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, with_brier: bool = True
) -> dict[str, float | int | None]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    both_classes = len(np.unique(labels)) > 1
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "pr_auc": (
            float(average_precision_score(labels, probabilities)) if both_classes else None
        ),
        "brier": (
            float(brier_score_loss(labels, probabilities)) if with_brier else None
        ),
    }


def _binary_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    return {
        "threshold": float(threshold),
        "recall": float(true_positive / positives) if positives else 0.0,
        "precision": (
            float(true_positive / (true_positive + false_positive))
            if true_positive + false_positive
            else 0.0
        ),
        "fpr": float(false_positive / negatives) if negatives else 0.0,
        "false_positive": false_positive,
        "true_positive": true_positive,
    }


def threshold_at_fpr(
    probabilities: np.ndarray, labels: np.ndarray, target_fpr: float
) -> dict[str, float | int]:
    """Pick the threshold with maximum recall subject to ``FPR <= target_fpr``."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = sorted({float(value) for value in probabilities} | {1.0})
    best: dict[str, float | int] | None = None
    for threshold in candidates:
        result = _binary_metrics(probabilities, labels, threshold)
        if float(result["fpr"]) > target_fpr:
            continue
        if best is None or (
            float(result["recall"]), float(result["threshold"])
        ) > (float(best["recall"]), float(best["threshold"])):
            best = result
    if best is None:
        best = _binary_metrics(probabilities, labels, 1.0)
    best["target_fpr"] = float(target_fpr)
    return best


def f1_optimal_threshold(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Return the threshold maximizing F1 on the given (validation) split."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best: dict[str, float | int] | None = None
    for threshold in sorted({float(value) for value in probabilities} | {1.0}):
        result = _binary_metrics(probabilities, labels, threshold)
        precision = float(result["precision"])
        recall = float(result["recall"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if best is None or f1 > float(best["f1"]):
            best = {**result, "f1": f1}
    assert best is not None
    return best


def serving_thresholds(
    valid_probabilities: np.ndarray, valid_labels: np.ndarray
) -> dict[str, Any]:
    """Derive suspicious/high-risk thresholds at 5%/1% validation FPR."""

    at_five = threshold_at_fpr(valid_probabilities, valid_labels, 0.05)
    at_one = threshold_at_fpr(valid_probabilities, valid_labels, 0.01)
    suspicious = float(at_five["threshold"])
    high_risk = float(at_one["threshold"])
    adjusted = False
    if high_risk <= suspicious:
        high_risk = min(1.0, suspicious + 0.05)
        adjusted = True
    if high_risk <= suspicious:
        suspicious = max(0.0, high_risk - 0.05)
        adjusted = True
    return {
        "suspicious": suspicious,
        "high_risk": high_risk,
        "adjusted_for_ordering": adjusted,
        "selected_on": "validation (player-disjoint)",
        "suspicious_validation": at_five,
        "high_risk_validation": at_one,
        "f1_optimal_validation": f1_optimal_threshold(valid_probabilities, valid_labels),
    }


def calibration_points(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Calibration curve points (uniform bins) for the test split."""

    observed, predicted = calibration_curve(
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        n_bins=n_bins,
        strategy="uniform",
    )
    return [
        {"predicted": float(prediction), "observed": float(observation)}
        for prediction, observation in zip(predicted, observed)
    ]


def _player_indices(groups: np.ndarray) -> dict[object, np.ndarray]:
    unique = np.unique(groups)
    return {
        player: np.flatnonzero(groups == player)
        for player in unique
    }


def bootstrap_by_player(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float | int | None]]:
    """Percentile CIs for test metrics, resampling whole players with replacement."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=object)
    indices = _player_indices(groups)
    players = np.asarray(list(indices), dtype=object)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "brier": [],
    }
    for tag in thresholds:
        samples[f"recall_at_{tag}"] = []
        samples[f"precision_at_{tag}"] = []
    for _ in range(iterations):
        drawn = rng.choice(players, size=len(players), replace=True)
        sample_index = np.concatenate([indices[player] for player in drawn])
        sample_probabilities = probabilities[sample_index]
        sample_labels = labels[sample_index]
        if len(np.unique(sample_labels)) < 2:
            continue
        metrics = _core_metrics(sample_probabilities, sample_labels)
        for metric in ("roc_auc", "pr_auc", "brier"):
            value = metrics[metric]
            if value is not None:
                samples[metric].append(float(value))
        for tag, threshold in thresholds.items():
            binary = _binary_metrics(sample_probabilities, sample_labels, threshold)
            samples[f"recall_at_{tag}"].append(float(binary["recall"]))
            samples[f"precision_at_{tag}"].append(float(binary["precision"]))

    point = _core_metrics(probabilities, labels)
    for tag, threshold in thresholds.items():
        binary = _binary_metrics(probabilities, labels, threshold)
        point[f"recall_at_{tag}"] = binary["recall"]
        point[f"precision_at_{tag}"] = binary["precision"]
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, values in samples.items():
        if not values:
            result[metric] = {"point": point.get(metric), "low": None, "high": None, "n": 0}
            continue
        result[metric] = {
            "point": point.get(metric),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "n": len(values),
        }
    result["players_in_test"] = {
        "point": int(len(players)),
        "low": None,
        "high": None,
        "n": len(players),
    }
    return result


def bootstrap_auc_difference(
    model_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | None]:
    """Paired player-level bootstrap of ``AUC(model) - AUC(baseline)``."""

    model_probabilities = np.asarray(model_probabilities, dtype=float)
    baseline_probabilities = np.asarray(baseline_probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=object)
    indices = _player_indices(groups)
    players = np.asarray(list(indices), dtype=object)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        drawn = rng.choice(players, size=len(players), replace=True)
        sample_index = np.concatenate([indices[player] for player in drawn])
        sample_labels = labels[sample_index]
        if len(np.unique(sample_labels)) < 2:
            continue
        model_auc = float(roc_auc_score(sample_labels, model_probabilities[sample_index]))
        baseline_auc = float(
            roc_auc_score(sample_labels, baseline_probabilities[sample_index])
        )
        deltas.append(model_auc - baseline_auc)
    if not deltas:
        return {"delta": None, "low": None, "high": None, "n": 0}
    return {
        "delta": float(np.mean(deltas)),
        "low": float(np.percentile(deltas, 2.5)),
        "high": float(np.percentile(deltas, 97.5)),
        "n": len(deltas),
    }


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _mean_by_label(
    values: Sequence[float], labels: Sequence[int]
) -> dict[str, float | int]:
    pairs = list(zip(values, labels))
    positives = [value for value, label in pairs if label == 1]
    negatives = [value for value, label in pairs if label == 0]
    return {
        "cheating": _mean(positives),
        "fair": _mean(negatives),
        "n_cheating": len(positives),
        "n_fair": len(negatives),
    }


def _baseline_metadata() -> Any:
    from packages.ml.contracts import MLModel, ModelMetadata, RiskThresholds

    return ModelMetadata(
        model_version=f"{MODEL_VERSION}-heuristic-baseline",
        model_type=MLModel.BASELINE,
        feature_schema_version="1.0.0",
        feature_names=list(FEATURE_NAMES),
        engine_name=ENGINE_NAME,
        engine_version=ENGINE_VERSION,
        analysis_depth=ANALYSIS_DEPTH,
        analysis_time_ms=1,
        thresholds=RiskThresholds(suspicious=0.45, high_risk=0.65),
        dataset_hash=None,
        trained_at=None,
    )


def _player_level_baseline(
    games: pd.DataFrame,
    test_game_ids: Sequence[str],
    model_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Evaluate BaselineScorer and the model at the player level on test players.

    ``BaselineScorer`` is a player-level heuristic with a documented minimum of
    10 games, so it is compared on players with at least 10 test games.  A
    player's target is the majority of their labelled games; the sample is small
    and the report says so instead of pretending it is powered.
    """

    from packages.feature_engineering.schema import GameAggregateFeatures
    from packages.ml.baseline import BaselineScorer, InsufficientGamesError
    from packages.ml.contracts import ScoringInput

    aggregate_columns = list(GameAggregateFeatures.model_fields)

    frame = games.copy()
    frame = frame.loc[frame["game_id"].isin(list(test_game_ids))].reset_index(drop=True)
    probability_by_game = {
        str(game_id): float(probability)
        for game_id, probability in zip(test_game_ids, model_probabilities)
    }
    label_by_game = {
        str(game_id): int(label) for game_id, label in zip(frame["game_id"], labels)
    }
    scorer = BaselineScorer(_baseline_metadata())
    baseline_scores: list[float] = []
    model_scores: list[float] = []
    player_labels: list[int] = []
    skipped = 0
    for player_id, player_games in frame.groupby("player_id", sort=True):
        if len(player_games) < scorer.minimum_games:
            skipped += 1
            continue
        aggregates = [
            game_from_record(row)
            for row in player_games.loc[:, aggregate_columns].to_dict(orient="records")
        ]
        try:
            result = scorer.score(
                ScoringInput(
                    username_normalized=str(player_id),
                    games=aggregates,
                )
            )
        except InsufficientGamesError:
            skipped += 1
            continue
        baseline_scores.append(float(result.risk_score))
        model_scores.append(
            _mean(
                probability_by_game[str(game_id)]
                for game_id in player_games["game_id"]
            )
        )
        game_labels = [label_by_game[str(game_id)] for game_id in player_games["game_id"]]
        player_labels.append(int(_mean(game_labels) >= 0.5))

    if not baseline_scores:
        return {
            "status": "insufficient_players",
            "minimum_games": scorer.minimum_games,
            "players_evaluated": 0,
            "players_skipped": skipped,
            "note": "no test player had enough games for the BaselineScorer contract",
        }
    both_classes = len(set(player_labels)) > 1
    return {
        "status": "evaluated",
        "minimum_games": scorer.minimum_games,
        "players_evaluated": len(baseline_scores),
        "players_skipped": skipped,
        "target": "player majority of game labels (>= 0.5)",
        "baseline_score_by_label": _mean_by_label(baseline_scores, player_labels),
        "model_score_by_label": _mean_by_label(model_scores, player_labels),
        "baseline_roc_auc": (
            float(roc_auc_score(player_labels, baseline_scores))
            if both_classes
            else None
        ),
        "model_roc_auc": (
            float(roc_auc_score(player_labels, model_scores)) if both_classes else None
        ),
        "underpowered_warning": (
            "player-level comparison uses a single small test split; treat it as "
            "a sanity check, not a powered benchmark"
        ),
    }


def elo_stratified_fpr(
    elos: Sequence[float],
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Report FPR/recall per Elo bin at a fixed operating threshold.

    Stratification is evaluation-only: no raw Elo enters the feature matrix.
    """

    frame = pd.DataFrame(
        {
            "elo": [float(value) for value in elos],
            "probability": np.asarray(probabilities, dtype=float),
            "label": np.asarray(labels, dtype=int),
        }
    )
    frame["bin"] = pd.cut(frame["elo"], bins=list(ELO_BIN_EDGES), right=False)
    bins: list[dict[str, Any]] = []
    for interval, group in frame.groupby("bin", observed=True, sort=True):
        predicted = group["probability"] >= threshold
        negatives = int((group["label"] == 0).sum())
        positives = int((group["label"] == 1).sum())
        false_positive = int((predicted & (group["label"] == 0)).sum())
        true_positive = int((predicted & (group["label"] == 1)).sum())
        bins.append(
            {
                "elo_bin": str(interval),
                "n": int(len(group)),
                "negatives": negatives,
                "positives": positives,
                "fpr": float(false_positive / negatives) if negatives else None,
                "recall": float(true_positive / positives) if positives else None,
            }
        )
    return {
        "threshold": float(threshold),
        "target_fpr": 0.05,
        "note": (
            "Elo is used only for evaluation stratification, never as a model "
            "feature; bins are small and estimates are correspondingly noisy."
        ),
        "bins": bins,
    }


def _heuristic_baselines(
    features: pd.DataFrame,
    valid_index: np.ndarray,
    test_index: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    scores = {
        "top_1_match_rate": features["top_1_match_rate"].to_numpy(dtype=float),
        "inverse_mean_centipawn_loss": -features["mean_centipawn_loss"].to_numpy(dtype=float),
    }
    results: dict[str, Any] = {}
    for name, values in scores.items():
        validation = threshold_at_fpr(values[valid_index], labels[valid_index], 0.05)
        threshold = float(validation["threshold"])
        metrics: dict[str, Any] = _core_metrics(
            values[test_index], labels[test_index], with_brier=False
        )
        metrics["at_fpr_5pct"] = _binary_metrics(
            values[test_index], labels[test_index], threshold
        )
        results[name] = {
            "validation_threshold_at_fpr_5pct": validation,
            "test": metrics,
        }
    return results


def per_move_research(
    raw: pd.DataFrame,
    *,
    seed: int = MODEL_SEED,
    folds: int = 3,
) -> dict[str, Any]:
    """Research-only per-move model with player-disjoint ``GroupKFold``.

    Uses the 12-feature contract from :mod:`packages.ml.feature_contract` on real
    ChessFraud rows.  It is reported for context and never exported to serve.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    from packages.ml.features import (
        build_training_features,
        fit_elo_centering,
    )

    try:
        eligible = raw.loc[
            raw["is_used"].eq(True) & (raw["half_move"] >= MIN_HALF_MOVE)
        ].reset_index(drop=True)
        labels = eligible["is_cheating_move"].to_numpy(dtype=int)
        groups = eligible["player_id"].to_numpy()
        predictions = np.full(len(eligible), np.nan, dtype=float)
        fold_reports: list[dict[str, float | int]] = []
        splitter = GroupKFold(n_splits=folds)
        for fold, (train_rows, test_rows) in enumerate(
            splitter.split(eligible, labels, groups)
        ):
            train_frame = eligible.iloc[train_rows].reset_index(drop=True)
            test_frame = eligible.iloc[test_rows].reset_index(drop=True)
            centering = fit_elo_centering(train_frame)
            train_features = build_training_features(
                train_frame, elo_centering=centering, allow_missing_timing=True
            )
            test_features = build_training_features(
                test_frame, elo_centering=centering, allow_missing_timing=True
            )
            validate_feature_columns(train_features.columns)
            validate_feature_columns(test_features.columns)
            fold_labels = test_frame["is_cheating_move"].to_numpy(dtype=int)
            if len(np.unique(fold_labels)) < 2:
                continue
            model: Any = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed),
            )
            model.fit(train_features.to_numpy(dtype=float), train_frame["is_cheating_move"])
            fold_probabilities = predict_probabilities(
                model, test_features.to_numpy(dtype=float)
            )
            predictions[test_rows] = fold_probabilities
            fold_reports.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train_rows)),
                    "test_rows": int(len(test_rows)),
                    "test_players": int(len(np.unique(groups[test_rows]))),
                    "roc_auc": float(roc_auc_score(fold_labels, fold_probabilities)),
                    "pr_auc": float(
                        average_precision_score(fold_labels, fold_probabilities)
                    ),
                }
            )
        complete = ~np.isnan(predictions)
        pooled_labels = labels[complete]
        return {
            "status": "completed",
            "purpose": "research only; not part of the serving artifact",
            "feature_names": list(FEATURE_NAMES),
            "split": f"GroupKFold(n_splits={folds}) by player_id",
            "rows": int(len(eligible)),
            "players": int(len(np.unique(groups))),
            "evaluated_rows": int(complete.sum()),
            "roc_auc_oof": (
                float(roc_auc_score(pooled_labels, predictions[complete]))
                if len(np.unique(pooled_labels)) > 1
                else None
            ),
            "pr_auc_oof": (
                float(average_precision_score(pooled_labels, predictions[complete]))
                if len(np.unique(pooled_labels)) > 1
                else None
            ),
            "folds": fold_reports,
            "caveat": (
                "Includes centered Elo features; the game-level serving model does "
                "not. Per-move AUC is not comparable to the game-level AUC."
            ),
        }
    except Exception as error:  # noqa: BLE001 - research section must not hide failures
        return {
            "status": "failed",
            "purpose": "research only; not part of the serving artifact",
            "error": f"{type(error).__name__}: {error}",
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    return value


def train(
    *,
    data_dir: str | Path | None = None,
    output_dir: str | Path = "data/models/chessfraud-v1",
    seed: int = MODEL_SEED,
    skip_per_move: bool = False,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Train the game-level artifact and return the full report payload."""

    started = time.perf_counter()
    output_path = Path(output_dir)
    parquet_path = resolve_data_path(CHESSFRAUD_FILENAME, data_dir=data_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"ChessFraud table not found: {parquet_path}")
    dataset_hash = sha256_file(parquet_path)
    raw = load_chessfraud(data_dir=data_dir)
    games = build_game_frame(raw)
    features_list, labels_list, groups_list, meta = game_dataset_from_frame(games)
    feature_frame = pd.DataFrame(features_list, columns=list(GAME_FEATURE_NAMES))
    assert_no_leakage(feature_frame.columns)
    validate_game_feature_names(GAME_FEATURE_NAMES)

    labels = np.asarray(labels_list, dtype=int)
    groups = np.asarray(groups_list, dtype=object)
    features = feature_frame.to_numpy(dtype=float)
    train_index, valid_index, test_index = split_by_player(groups, seed=seed)
    test_game_ids = meta.loc[test_index, "game_id"].tolist()
    elo_by_game = (
        raw.loc[raw["is_used"].eq(True), ["game_id", "player_elo"]]
        .drop_duplicates("game_id")
        .set_index("game_id")["player_elo"]
    )
    test_elos = [float(elo_by_game[str(game_id)]) for game_id in test_game_ids]
    if len(np.unique(labels[train_index])) < 2 or len(np.unique(labels[valid_index])) < 2:
        raise ValueError(
            "player-disjoint split produced a single-class train/validation set; "
            "collect more players or lower the split granularity"
        )

    candidates: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    for name, estimator in _estimators(seed).items():
        calibrated = _fit_calibrated(
            estimator,
            features[train_index],
            labels[train_index],
            features[valid_index],
            labels[valid_index],
        )
        models[name] = calibrated
        validation_probabilities[name] = predict_probabilities(
            calibrated, features[valid_index]
        )
        test_probabilities[name] = predict_probabilities(calibrated, features[test_index])
        candidates[name] = {
            "validation": {
                **_core_metrics(validation_probabilities[name], labels[valid_index]),
                "thresholds": serving_thresholds(
                    validation_probabilities[name], labels[valid_index]
                ),
            },
            "test": _core_metrics(test_probabilities[name], labels[test_index]),
            "feature_importance": feature_importance(calibrated, GAME_FEATURE_NAMES),
        }

    serving_model = models[SERVING_MODEL]
    serving_validation = validation_probabilities[SERVING_MODEL]
    serving_test = test_probabilities[SERVING_MODEL]
    thresholds = serving_thresholds(serving_validation, labels[valid_index])
    threshold_map = {
        "fpr_1pct": float(thresholds["high_risk"]),
        "fpr_5pct": float(thresholds["suspicious"]),
    }
    test_metrics: dict[str, Any] = _core_metrics(serving_test, labels[test_index])
    test_metrics["at_fpr_1pct"] = _binary_metrics(
        serving_test, labels[test_index], threshold_map["fpr_1pct"]
    )
    test_metrics["at_fpr_5pct"] = _binary_metrics(
        serving_test, labels[test_index], threshold_map["fpr_5pct"]
    )
    bootstrap = bootstrap_by_player(
        serving_test,
        labels[test_index],
        groups[test_index],
        threshold_map,
        iterations=bootstrap_iterations,
    )

    baselines = _heuristic_baselines(
        feature_frame, valid_index, test_index, labels
    )
    heuristic_aucs = {
        name: float(payload["test"]["roc_auc"])
        for name, payload in baselines.items()
        if payload["test"]["roc_auc"] is not None
    }
    if heuristic_aucs:
        best_heuristic = max(heuristic_aucs, key=heuristic_aucs.__getitem__)
        heuristic_scores = (
            feature_frame["top_1_match_rate"].to_numpy(dtype=float)
            if best_heuristic == "top_1_match_rate"
            else -feature_frame["mean_centipawn_loss"].to_numpy(dtype=float)
        )
        difference = bootstrap_auc_difference(
            serving_test,
            heuristic_scores[test_index],
            labels[test_index],
            groups[test_index],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        delta_low = difference["low"]
        adds_value = bool(delta_low is not None and float(delta_low) > 0)
        comparison = {
            "best_heuristic": best_heuristic,
            "model_test_roc_auc": test_metrics["roc_auc"],
            "heuristic_test_roc_auc": heuristic_aucs[best_heuristic],
            "paired_player_bootstrap_delta_auc": difference,
            "verdict": (
                "trained model adds value over the heuristic"
                if adds_value
                else "value over the heuristic is NOT demonstrated on this test split"
            ),
        }
    else:
        comparison = {
            "best_heuristic": None,
            "verdict": "heuristic AUC could not be computed on the test split",
        }

    baseline_scorer = _player_level_baseline(
        games,
        test_game_ids,
        serving_test,
        labels[test_index],
        groups[test_index],
    )
    baseline_scorer["verdict"] = (
        "not a powered comparison: player-level AUC is computed on a single small "
        "test split with very few players, so it can only suggest direction. The "
        "game-level heuristic comparison above is the statistical check."
    )

    importance = feature_importance(serving_model, GAME_FEATURE_NAMES)
    feature_baselines = {
        name: float(feature_frame[name].mean()) for name in GAME_FEATURE_NAMES
    }
    trained_at = datetime.now(UTC).date().isoformat()
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "model_type": SERVING_MODEL,
        "feature_schema_version": GAME_FEATURE_SCHEMA_VERSION,
        "feature_names": list(GAME_FEATURE_NAMES),
        "engine_name": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "analysis_depth": ANALYSIS_DEPTH,
        "analysis_time_ms": 1,
        "analysis_time_ms_semantics": (
            "ChessFraud uses fixed depth 15 with no time cap; the shared metadata "
            "contract requires analysis_time_ms >= 1, so 1 is a sentinel meaning "
            "fixed-depth/no time limit."
        ),
        "trained_at": trained_at,
        "dataset_hash": dataset_hash,
        "dataset": {
            "path_name": CHESSFRAUD_FILENAME,
            "raw_rows": int(len(raw)),
            "used_rows": int(raw["is_used"].eq(True).sum()),
            "player_games": int(len(games)),
            "players": int(len(np.unique(groups))),
            "player_game_prevalence": float(labels.mean()),
        },
        "thresholds": {
            "suspicious": float(thresholds["suspicious"]),
            "high_risk": float(thresholds["high_risk"]),
        },
        "threshold_selection": {
            key: value
            for key, value in thresholds.items()
            if key not in {"suspicious", "high_risk"}
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "fitted_on": "validation split (player-disjoint), frozen base estimator",
            "base_estimator": SERVING_MODEL,
            "calibration_rows": int(len(valid_index)),
        },
        "global_feature_importance": importance,
        "feature_baselines": feature_baselines,
        "split_counts": {
            "train_games": int(len(train_index)),
            "validation_games": int(len(valid_index)),
            "test_games": int(len(test_index)),
            "train_players": int(len(np.unique(groups[train_index]))),
            "validation_players": int(len(np.unique(groups[valid_index]))),
            "test_players": int(len(np.unique(groups[test_index]))),
            "train_prevalence": float(labels[train_index].mean()),
            "validation_prevalence": float(labels[valid_index].mean()),
            "test_prevalence": float(labels[test_index].mean()),
        },
        "prevalence_caveat": PREVALENCE_CAVEAT,
    }

    report: dict[str, Any] = {
        "report_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "split": {
            "strategy": "GroupShuffleSplit by player_id, 60/20/20, player-disjoint",
            "counts": metadata["split_counts"],
        },
        "candidate_models": candidates,
        "selected_model": SERVING_MODEL,
        "selection_rationale": (
            "XGBoost is the declared serving model; the logistic regression is an "
            "interpretable comparison. Neither was selected using test metrics."
        ),
        "serving_choice_note": {
            "why_xgboost": (
                "the shared MLModel contract enumerates only baseline and xgboost, "
                "and xgboost is the declared serving model in the plan"
            ),
            "test_only_observation": (
                "On this test split the logistic regression scored higher "
                f"ROC-AUC ({candidates['logreg']['test']['roc_auc']:.4f}) than xgboost "
                f"({candidates['xgboost']['test']['roc_auc']:.4f}). This is a test-set "
                "observation and was not used for model selection; choosing it now "
                "would tune on test. A future model version can compare candidates "
                "on validation PR-AUC and amend the contract."
            ),
        },
        "test": {
            "metrics": test_metrics,
            "bootstrap_by_player": bootstrap,
            "calibration_curve": calibration_points(serving_test, labels[test_index]),
            "elo_stratified_fpr": elo_stratified_fpr(
                test_elos,
                serving_test,
                labels[test_index],
                float(thresholds["suspicious"]),
            ),
        },
        "thresholds": metadata["thresholds"],
        "threshold_selection": metadata["threshold_selection"],
        "baselines": {
            "heuristics": baselines,
            "baseline_scorer_player_level": baseline_scorer,
        },
        "adds_value": comparison,
        "leakage_gate": {
            "feature_columns": list(GAME_FEATURE_NAMES),
            "forbidden_columns_absent": True,
            "player_splits_disjoint": True,
            "dataset_hash_recorded": dataset_hash,
            "per_move_contract_columns_checked_in_research": list(FEATURE_NAMES),
        },
        "global_feature_importance": importance,
        "prevalence_caveat": PREVALENCE_CAVEAT,
        "per_move_research": (
            {"status": "skipped", "reason": "--skip-per-move"}
            if skip_per_move
            else per_move_research(raw, seed=seed)
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(serving_model, output_path / "model.joblib")
    (output_path / "metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "training_report.json").write_text(
        json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory with chessfraud_full.parquet (default: FAIR_PLAY_DATA_DIR or data_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/chessfraud-v1"),
        help="artifact directory (model.joblib, metadata.json, training_report.json)",
    )
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument(
        "--skip-per-move",
        action="store_true",
        help="skip the research-only per-move GroupKFold section",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_per_move=bool(args.skip_per_move),
    )
    summary = {
        "model_version": report["model_version"],
        "dataset_hash": report["dataset_hash"],
        "split": report["split"]["counts"],
        "test": report["test"]["metrics"],
        "baselines": report["baselines"],
        "adds_value": report["adds_value"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
