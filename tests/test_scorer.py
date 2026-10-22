"""Tests for the trained game-level scorer artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

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


class _MeanCplModel:
    """Deterministic test double: probability = mean_centipawn_loss / 100."""

    def predict_proba(self, features: object) -> list[list[float]]:
        rows = []
        for row in features:  # type: ignore[union-attr]
            probability = min(1.0, max(0.0, float(row[0]) / 100.0))
            rows.append([1.0 - probability, probability])
        return rows


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    model = _MeanCplModel()
    importances = [
        1.0 / (index + 1) for index in range(len(GAME_FEATURE_NAMES))
    ]
    total = sum(importances)
    baselines = {name: 20.0 for name in GAME_FEATURE_NAMES}
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
        "player_thresholds": {
            "suspicious": 0.25,
            "high_risk": 0.45,
            "derivation": "player mean + 1/2 population std on validation",
            "validation_player_mean": 0.2,
            "validation_player_std": 0.05,
            "validation_players": 10,
            "note": "small-N; recalibrate on a larger cohort",
        },
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
    # ModelMetadata keeps the per-game contract for reporting.
    assert result.model.thresholds.suspicious == 0.4
    assert result.model.thresholds.high_risk == 0.7
    # Verdicts use the player-mean thresholds from the artifact.
    assert scorer.player_thresholds.suspicious == 0.25
    assert scorer.player_thresholds.high_risk == 0.45
    assert scorer.artifact_metadata["player_thresholds"]["validation_players"] == 10
    assert scorer.artifact_metadata["player_thresholds"]["derivation"]

    thresholds = scorer.player_thresholds
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


@pytest.mark.parametrize(
    ("mean_centipawn_loss", "expected_level"),
    [
        # 0.30 is below the old per-game suspicious threshold (0.40) but above
        # the player-level suspicious threshold (0.25).
        (30.0, "suspicious"),
        # 0.50 is below the per-game high-risk threshold (0.70) but above the
        # player-level high-risk threshold (0.45).
        (50.0, "high_risk"),
        (10.0, "low_risk"),
    ],
)
def test_player_thresholds_drive_risk_level(
    artifact_dir: Path,
    mean_centipawn_loss: float,
    expected_level: str,
) -> None:
    scorer = TrainedScorer.load(artifact_dir)
    games = [
        _game(mean_centipawn_loss=mean_centipawn_loss) for _ in range(12)
    ]

    result = scorer.score(ScoringInput(username_normalized="tester", games=games))

    assert result.risk_score == pytest.approx(mean_centipawn_loss / 100.0)
    if expected_level != "low_risk":
        # Regression check for the bug: these player means never reach the
        # per-game high-risk threshold, so the old code always said low_risk.
        assert result.risk_score < result.model.thresholds.high_risk
    if expected_level == "suspicious":
        assert result.risk_score < result.model.thresholds.suspicious
    assert result.risk_level == expected_level


def test_player_threshold_metadata_round_trip(artifact_dir: Path) -> None:
    scorer = TrainedScorer.load(artifact_dir)
    artifact = scorer.artifact_metadata["player_thresholds"]

    assert artifact["suspicious"] == pytest.approx(0.25)
    assert artifact["high_risk"] == pytest.approx(0.45)
    assert artifact["validation_player_mean"] == pytest.approx(0.2)
    assert artifact["validation_player_std"] == pytest.approx(0.05)
    assert artifact["note"] == "small-N; recalibrate on a larger cohort"


def test_load_falls_back_to_per_game_thresholds(
    artifact_dir: Path, tmp_path: Path
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    joblib.dump(joblib.load(artifact_dir / "model.joblib"), legacy / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    metadata.pop("player_thresholds")
    (legacy / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    scorer = TrainedScorer.load(legacy)

    assert scorer.player_thresholds.suspicious == scorer.metadata.thresholds.suspicious
    assert scorer.player_thresholds.high_risk == scorer.metadata.thresholds.high_risk


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
