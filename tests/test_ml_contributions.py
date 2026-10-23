"""Per-verdict contribution tests for the trained and baseline scorers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from packages.feature_engineering.schema import GameAggregateFeatures
from packages.ml.baseline import BaselineScorer
from packages.ml.contracts import (
    ContributionDirection,
    ModelMetadata,
    RiskThresholds,
    ScoringInput,
)
from packages.ml.game_features import GAME_FEATURE_NAMES
from packages.ml.scorer import TrainedScorer


def _metadata(model_type: str = "xgboost") -> ModelMetadata:
    return ModelMetadata(
        model_version="contrib-test",
        model_type=model_type,  # type: ignore[arg-type]
        feature_schema_version="1.0.0",
        feature_names=list(GAME_FEATURE_NAMES),
        engine_name="stockfish",
        engine_version="17",
        analysis_depth=15,
        analysis_time_ms=1,
        thresholds=RiskThresholds(suspicious=0.4, high_risk=0.7),
    )


def _game(**overrides: object) -> GameAggregateFeatures:
    values: dict[str, object] = {
        "move_count": 30,
        "mean_centipawn_loss": 20.0,
        "median_centipawn_loss": 12.0,
        "p90_centipawn_loss": 60.0,
        "top_1_match_rate": 0.8,
        "top_3_match_rate": 0.9,
        "difficult_position_match_rate": 0.8,
        "tactical_error_count": 2,
        "centipawn_loss_stability": 120.0,
        "identical_time_share": 0.4,
        "median_time_seconds": 4.0,
        "mean_time_seconds": 5.0,
        "time_variance": 11.0,
        "time_difficulty_correlation": None,
        "maia_divergence": 0.2,
    }
    values.update(overrides)
    return GameAggregateFeatures.model_validate(values)


def _trained_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2025)
    features = rng.normal(size=(80, len(GAME_FEATURE_NAMES)))
    labels = (features[:, 3] + 0.4 * rng.normal(size=80) > 0.0).astype(int)
    labels[0], labels[1] = 0, 1
    return features, labels


def _xgboost_model() -> XGBClassifier:
    features, labels = _trained_data()
    model = XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.2,
        n_jobs=1,
        random_state=7,
    )
    model.fit(features, labels)
    return model


def _scorer(model: Any) -> TrainedScorer:
    importances = [
        (name, 1.0 / (index + 1)) for index, name in enumerate(GAME_FEATURE_NAMES)
    ]
    return TrainedScorer(
        model,
        _metadata(),
        feature_importances=importances,
        feature_baselines={name: 0.0 for name in GAME_FEATURE_NAMES},
        artifact_metadata={},
    )


def _games(count: int = 12) -> list[GameAggregateFeatures]:
    return [
        _game(
            mean_centipawn_loss=18.0 + index,
            top_1_match_rate=0.7 + 0.01 * index,
        )
        for index in range(count)
    ]


def _assert_consistent(evidence) -> None:
    for item in evidence:
        assert item.contribution is not None
        assert math.isfinite(item.contribution)
        assert item.contribution_share is not None
        assert 0.0 <= item.contribution_share <= 1.0
        expected = (
            ContributionDirection.SUPPORTS_RISK
            if item.contribution > 1e-12
            else ContributionDirection.REDUCES_RISK
            if item.contribution < -1e-12
            else ContributionDirection.NEUTRAL
        )
        assert item.direction is expected


def test_trained_scorer_emits_finite_contributions_in_evidence_order() -> None:
    scorer = _scorer(_xgboost_model())
    result = scorer.score(
        ScoringInput(username_normalized="tester", games=_games())
    )

    assert result.evidence
    _assert_consistent(result.evidence)
    assert all(
        item.code == f"GAME_{name.upper()}"
        for item, name in zip(
            result.evidence, GAME_FEATURE_NAMES[: len(result.evidence)], strict=True
        )
    )
    total = sum(abs(item.contribution_share or 0.0) for item in result.evidence)
    assert total <= 1.0 + 1e-9


def test_calibrated_wrapped_xgboost_exposes_contributions() -> None:
    features, labels = _trained_data()
    base = _xgboost_model()
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(features, labels)

    scorer = _scorer(calibrated)
    result = scorer.score(
        ScoringInput(username_normalized="tester", games=_games())
    )

    _assert_consistent(result.evidence)


def test_non_xgboost_model_keeps_global_importance_only() -> None:
    class DummyModel:
        def predict_proba(self, rows: Any) -> list[list[float]]:
            return [[0.4, 0.6] for _ in rows]

    scorer = _scorer(DummyModel())
    result = scorer.score(
        ScoringInput(username_normalized="tester", games=_games())
    )

    assert result.evidence
    assert all(item.contribution is None for item in result.evidence)
    assert all(item.contribution_share is None for item in result.evidence)
    assert all(item.direction is None for item in result.evidence)


def test_baseline_contributions_are_signed_deltas_and_do_not_change_score() -> None:
    scorer = BaselineScorer(_metadata("baseline"))
    games = [_game() for _ in range(12)]
    result = scorer.score(
        ScoringInput(username_normalized="tester", games=games)
    )

    expected_score = 0.35 * 0.8 + 0.30 * (1.0 - 20.0 / 100.0) + 0.20 * 0.4 + 0.15 * 0.2
    assert result.risk_score == pytest.approx(expected_score)

    by_code = {item.code: item for item in result.evidence}
    assert by_code["HIGH_TOP_MOVE_RATE"].contribution == pytest.approx(
        0.35 * (0.8 - 0.61)
    )
    assert by_code["LOW_CENTIPAWN_LOSS"].contribution == pytest.approx(
        0.30 * ((1.0 - 20.0 / 100.0) - (1.0 - 35.0 / 100.0))
    )
    assert by_code["REGULAR_MOVE_TIMES"].contribution == pytest.approx(
        0.20 * (0.4 - 0.20)
    )
    assert by_code["MAIA_DIVERGENCE"].contribution == pytest.approx(
        0.15 * (0.2 - 0.15)
    )
    _assert_consistent(result.evidence)
    total = sum(abs(item.contribution_share or 0.0) for item in result.evidence)
    assert total == pytest.approx(1.0)
