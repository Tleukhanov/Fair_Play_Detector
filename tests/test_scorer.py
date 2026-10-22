"""Tests for the trained game-level scorer artifact adapter."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import pytest
from xgboost import XGBClassifier

from packages.feature_engineering.schema import GameAggregateFeatures
from packages.ml.baseline import InsufficientGamesError
from packages.ml.contracts import ScoringInput
from packages.ml.feature_contract import FeatureContractError
from packages.ml.game_features import (
    GAME_FEATURE_NAMES,
    game_feature_values,
    game_feature_vector,
)
from packages.ml.scorer import TrainedScorer


def _game(**overrides: object) -> GameAggregateFeatures:
    values: dict[str, object] = {
        "move_count": 30,
        "mean_centipawn_loss": 20.0,
        "median_centipawn_loss": 12.0,
        "p90_centipawn_loss": 60.0,
        "top_1_match_rate": 0.5,
        "top_3_match_rate": 0.7,
        "difficult_position_match_rate": 0.5,
        "tactical_error_count": 2,
        "centipawn_loss_stability": 120.0,
        "identical_time_share": 0.1,
        "median_time_seconds": 4.0,
        "mean_time_seconds": 5.0,
        "time_variance": 11.0,
        "time_difficulty_correlation": None,
        "maia_divergence": None,
    }
    values.update(overrides)
    return GameAggregateFeatures.model_validate(values)


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    rng = random.Random(11)
    features = [
        [rng.uniform(0, 120) for _ in GAME_FEATURE_NAMES] for _ in range(80)
    ]
    labels = [index % 2 for index in range(len(features))]
    model = XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.1,
        n_jobs=1,
        random_state=0,
    )
    model.fit(features, labels)

    importances = [float(value) for value in model.feature_importances_]
    total = sum(importances) or 1.0
    baselines = {
        name: sum(row[index] for row in features) / len(features)
        for index, name in enumerate(GAME_FEATURE_NAMES)
    }
    metadata = {
        "model_version": "chessfraud-v1-test",
        "model_type": "xgboost",
        "feature_schema_version": "1.0.0",
        "feature_names": list(GAME_FEATURE_NAMES),
        "engine_name": "stockfish",
        "engine_version": "17",
        "analysis_depth": 15,
        "analysis_time_ms": 1,
        "trained_at": "2025-01-01",
        "dataset_hash": "a" * 64,
        "thresholds": {"suspicious": 0.4, "high_risk": 0.7},
        "global_feature_importance": [
            {"name": name, "importance": value / total}
            for name, value in zip(GAME_FEATURE_NAMES, importances)
        ],
        "feature_baselines": baselines,
        "prevalence_caveat": "prevalence in ChessFraud is not deployment prevalence",
    }
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    joblib.dump(model, artifact / "model.joblib")
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return artifact


def test_score_implements_contract_and_grounded_evidence(
    artifact_dir: Path,
) -> None:
    scorer = TrainedScorer.load(artifact_dir)
    games = [
        _game(
            mean_centipawn_loss=10.0 + index,
            top_1_match_rate=0.4 + 0.02 * index,
            median_time_seconds=3.0 + index,
            mean_time_seconds=4.0 + index,
            time_variance=5.0 + index,
        )
        for index in range(12)
    ]

    result = scorer.score(ScoringInput(username_normalized="tester", games=games))

    assert 0.0 <= result.risk_score <= 1.0
    lower, upper = result.confidence_interval
    assert 0.0 <= lower <= result.risk_score <= upper <= 1.0
    assert result.games_analyzed == 12
    assert result.model.model_version == "chessfraud-v1-test"
    assert result.model.feature_names == list(GAME_FEATURE_NAMES)
    assert result.model.thresholds.suspicious == 0.4
    assert result.model.thresholds.high_risk == 0.7

    thresholds = result.model.thresholds
    expected_level = (
        "high_risk"
        if result.risk_score >= thresholds.high_risk
        else "suspicious"
        if result.risk_score >= thresholds.suspicious
        else "low_risk"
    )
    assert result.risk_level == expected_level

    assert 1 <= len(result.evidence) <= 5
    top_importance = {
        item["name"]
        for item in json.loads(
            (artifact_dir / "metadata.json").read_text()
        )["global_feature_importance"][:5]
    }
    for item in result.evidence:
        assert item.code.startswith("GAME_")
        assert item.code.removeprefix("GAME_").lower() in GAME_FEATURE_NAMES
        assert item.weight >= 0.0
        name = item.code.removeprefix("GAME_").lower()
        assert name in top_importance
        expected_mean = sum(
            game_feature_values(game)[name] for game in games
        ) / len(games)
        assert item.value == pytest.approx(expected_mean)
        assert item.baseline >= 0.0


def test_missing_timing_is_zero_filled_with_flag(artifact_dir: Path) -> None:
    scorer = TrainedScorer.load(artifact_dir)
    games = [
        _game(
            median_time_seconds=None,
            mean_time_seconds=None,
            time_variance=None,
        )
        for _ in range(10)
    ]

    result = scorer.score(ScoringInput(username_normalized="no-clocks", games=games))

    assert result.games_analyzed == 10
    vector = game_feature_vector(games[0])
    values = dict(zip(GAME_FEATURE_NAMES, vector))
    assert values["has_timing"] == 0.0
    assert values["median_time_seconds"] == 0.0
    assert values["mean_time_seconds"] == 0.0
    assert values["time_variance"] == 0.0


def test_insufficient_games_raises(artifact_dir: Path) -> None:
    scorer = TrainedScorer.load(artifact_dir)
    games = [_game() for _ in range(scorer.minimum_games - 1)]

    with pytest.raises(InsufficientGamesError):
        scorer.score(ScoringInput(username_normalized="tester", games=games))


def test_vector_length_and_order() -> None:
    game = _game()
    vector = game_feature_vector(game)
    values = game_feature_values(game)

    assert len(vector) == len(GAME_FEATURE_NAMES) == 11
    assert vector == [values[name] for name in GAME_FEATURE_NAMES]
    assert GAME_FEATURE_NAMES[0] == "mean_centipawn_loss"
    assert GAME_FEATURE_NAMES[-1] == "has_timing"


def test_load_rejects_unknown_feature_contract(
    artifact_dir: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    joblib.dump(joblib.load(artifact_dir / "model.joblib"), broken / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    metadata["feature_names"][0] = "player_elo"
    (broken / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FeatureContractError):
        TrainedScorer.load(broken)
