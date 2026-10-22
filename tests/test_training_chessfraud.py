"""Tests for the ChessFraud game-level training CLI and its gates."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from packages.ml.feature_contract import FeatureContractError
from packages.ml.game_features import GAME_FEATURE_NAMES
from packages.ml.scorer import TrainedScorer
from packages.ml.train import assert_no_leakage, sha256_file, split_by_player, train


def _synthetic_rows(
    *, players: int = 24, games_per_player: int = 5, moves_per_game: int = 8
) -> list[dict[str, object]]:
    rng = random.Random(7)
    rows: list[dict[str, object]] = []
    for player in range(players):
        cheating = player % 2 == 0
        for game in range(games_per_player):
            game_id = f"game-{player:02d}-{game:02d}"
            for move in range(moves_per_game):
                is_top = rng.random() < (0.8 if cheating else 0.35)
                loss = rng.randint(0, 20) if cheating else rng.randint(10, 120)
                played = "e2e4" if is_top else "a2a3"
                rows.append(
                    {
                        "tournament_id": 1,
                        "game_id": game_id,
                        "player_id": player,
                        "opponent_id": 1000 + player,
                        "player_color": "white",
                        "player_elo": 1500 + player,
                        "opponent_elo": 1480,
                        "time_control": "300+0",
                        "game_result": "1-0",
                        "half_move": 21 + move,
                        "move_player": played,
                        "position_before": "8/8/8/8/8/8/8/8 w - - 0 1",
                        "position_after": "8/8/8/8/8/8/8/8 b - - 0 1",
                        "move_thinking_time": 2.0 + rng.random(),
                        "clock_remaining_time": 200.0,
                        "is_used": True,
                        "eval_before": 40,
                        "eval_after": 40 - loss,
                        "centipawn_loss": loss,
                        "normalized_centipawn_loss": 0.1,
                        "move_stockfish_1": played if is_top else "h2h3",
                        "move_stockfish_9": played if is_top else "h2h3",
                        "move_stockfish_15": played if is_top else "h2h3",
                        "move_maia2_2050": "h2h3",
                        "maia2_win_prob_2050": 0.5,
                        "move_allie_2500": "h2h3",
                        "allie_win_prob_2500": 0.5,
                        "player_hint_shown": False,
                        "assistance_search_depth": 0,
                        "assistance_line_rank": 1,
                        "is_cheating_move": cheating,
                        "is_cheating_player_game": cheating,
                        "is_accused_by_opponent": False,
                    }
                )
    return rows


def _write_dataset(directory: Path) -> Path:
    rows = _synthetic_rows()
    parquet.write_table(
        pa.Table.from_pylist(rows), directory / "chessfraud_full.parquet"
    )
    return directory


def test_split_by_player_is_disjoint_and_covers_all_rows() -> None:
    groups = [f"player-{index % 30}" for index in range(150)]
    train_index, valid_index, test_index = split_by_player(groups, seed=2025)

    train_players = {groups[index] for index in train_index}
    valid_players = {groups[index] for index in valid_index}
    test_players = {groups[index] for index in test_index}
    assert not train_players & valid_players
    assert not train_players & test_players
    assert not valid_players & test_players
    assert len(train_index) + len(valid_index) + len(test_index) == len(groups)
    assert 0.5 < len(train_index) / len(groups) < 0.75


def test_training_writes_round_trippable_artifact(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path)
    output_dir = tmp_path / "model"

    report = train(
        data_dir=data_dir,
        output_dir=output_dir,
        seed=2025,
        skip_per_move=True,
        bootstrap_iterations=25,
    )

    for filename in ("model.joblib", "metadata.json", "training_report.json"):
        assert (output_dir / filename).is_file()

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["model_version"] == "chessfraud-v1"
    assert metadata["model_type"] == "xgboost"
    assert metadata["feature_names"] == list(GAME_FEATURE_NAMES)
    assert metadata["feature_schema_version"] == "1.0.0"
    assert metadata["engine_name"] == "stockfish"
    assert metadata["engine_version"] == "17"
    assert metadata["analysis_depth"] == 15
    assert metadata["dataset_hash"] == sha256_file(
        data_dir / "chessfraud_full.parquet"
    )
    assert len(metadata["dataset_hash"]) == 64
    assert metadata["thresholds"]["high_risk"] > metadata["thresholds"]["suspicious"]
    assert metadata["calibration"]["fitted_on"].startswith("validation")
    assert metadata["global_feature_importance"]
    assert metadata["prevalence_caveat"]
    assert metadata["split_counts"]["train_games"] > 0

    scorer = TrainedScorer.load(output_dir)
    assert scorer.metadata.feature_names == list(GAME_FEATURE_NAMES)
    assert scorer.metadata.dataset_hash == metadata["dataset_hash"]

    assert report["leakage_gate"]["player_splits_disjoint"] is True
    assert report["per_move_research"]["status"] == "skipped"
    assert report["seed"] == 2025


def test_report_contains_metrics_baselines_and_calibration(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path)
    report = train(
        data_dir=data_dir,
        output_dir=tmp_path / "model",
        skip_per_move=True,
        bootstrap_iterations=25,
    )

    metrics = report["test"]["metrics"]
    for key in ("roc_auc", "pr_auc", "brier", "prevalence"):
        assert key in metrics
    for key in ("at_fpr_1pct", "at_fpr_5pct"):
        assert {"recall", "precision", "fpr", "threshold"} <= set(metrics[key])

    bootstrap = report["test"]["bootstrap_by_player"]
    assert bootstrap["roc_auc"]["point"] == pytest.approx(metrics["roc_auc"])
    assert bootstrap["roc_auc"]["low"] <= bootstrap["roc_auc"]["high"]
    assert report["test"]["calibration_curve"]
    elo_report = report["test"]["elo_stratified_fpr"]
    assert elo_report["target_fpr"] == 0.05
    assert elo_report["bins"]
    assert all("fpr" in item and "recall" in item for item in elo_report["bins"])

    baselines = report["baselines"]["heuristics"]
    assert set(baselines) == {"top_1_match_rate", "inverse_mean_centipawn_loss"}
    assert "baseline_scorer_player_level" in report["baselines"]
    assert report["adds_value"]["verdict"]
    assert set(report["candidate_models"]) == {"xgboost", "logreg"}
    assert report["threshold_selection"]["selected_on"].startswith("validation")
    assert report["threshold_selection"]["f1_optimal_validation"]["f1"] >= 0.0


def test_leakage_gate_rejects_forbidden_columns() -> None:
    assert_no_leakage(GAME_FEATURE_NAMES)

    with pytest.raises(FeatureContractError):
        assert_no_leakage([*GAME_FEATURE_NAMES, "player_elo"])
    with pytest.raises(FeatureContractError):
        assert_no_leakage([*GAME_FEATURE_NAMES, "is_cheating_player_game"])
    with pytest.raises(FeatureContractError):
        assert_no_leakage([*GAME_FEATURE_NAMES, "mean_centipawn_loss_raw_elo"])
    with pytest.raises(FeatureContractError):
        assert_no_leakage(GAME_FEATURE_NAMES[:-1])
