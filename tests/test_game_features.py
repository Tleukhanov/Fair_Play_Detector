"""Tests for the game-level ChessFraud feature builder."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.chess_analysis.features import aggregate_game_features
from packages.ml.feature_contract import FORBIDDEN_COLUMNS, FeatureContractError
from packages.ml.game_features import (
    GAME_FEATURE_NAMES,
    GAME_METADATA_COLUMNS,
    GameFeatureError,
    assert_game_feature_contract,
    build_game_dataset,
    build_game_frame,
    game_feature_values,
    game_feature_vector,
    game_from_record,
    move_analysis_from_row,
    validate_game_feature_names,
)


def _row(
    game_id: str,
    player_id: int,
    half_move: int,
    *,
    label: bool = False,
    is_used: bool = True,
    best_move: str = "e2e4",
    played_move: str = "e2e4",
    centipawn_loss: int = 10,
    thinking: float = 3.0,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "player_id": player_id,
        "is_used": is_used,
        "half_move": half_move,
        "position_before": "8/8/8/8/8/8/8/8 w - - 0 1",
        "move_player": played_move,
        "move_thinking_time": thinking,
        "eval_before": 30,
        "eval_after": 20,
        "centipawn_loss": centipawn_loss,
        "is_cheating_player_game": label,
        "move_stockfish_1": best_move,
        "move_stockfish_9": best_move,
        "move_stockfish_15": best_move,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_build_game_frame_groups_by_game_and_filters_unused_rows() -> None:
    rows = [
        _row("game-a", 1, 10, label=True),  # opening: filtered
        _row("game-a", 1, 21, label=True, centipawn_loss=10),
        _row("game-a", 1, 22, label=True, centipawn_loss=30),
        _row("game-b", 2, 21, label=False, is_used=False),  # unused: filtered
        _row("game-b", 2, 22, label=False, centipawn_loss=50, thinking=4.0),
        _row("game-b", 2, 23, label=False, centipawn_loss=50, thinking=4.0),
    ]

    games = build_game_frame(_frame(rows))

    assert len(games) == 2
    assert list(games["game_id"]) == ["game-a", "game-b"]
    assert list(games["player_id"]) == [1, 2]
    assert list(games["label"]) == [1, 0]
    assert set(GAME_METADATA_COLUMNS).issubset(games.columns)
    assert set(GAME_FEATURE_NAMES).issubset(games.columns)

    first = games.iloc[0]
    assert first["move_count"] == 2
    assert first["mean_centipawn_loss"] == pytest.approx(20.0)
    assert first["top_1_match_rate"] == pytest.approx(1.0)
    assert first["tactical_error_count"] == 0


def test_game_frame_matches_direct_aggregation() -> None:
    rows = [
        _row("g1", 7, 21, label=True, centipawn_loss=0),
        _row("g1", 7, 22, label=True, centipawn_loss=120),
        _row("g1", 7, 23, label=True, centipawn_loss=40, played_move="a2a3"),
    ]
    games = build_game_frame(_frame(rows))
    expected = aggregate_game_features([move_analysis_from_row(row) for row in rows])

    assert games.iloc[0]["mean_centipawn_loss"] == pytest.approx(
        expected.mean_centipawn_loss
    )
    assert games.iloc[0]["p90_centipawn_loss"] == pytest.approx(
        expected.p90_centipawn_loss
    )
    assert games.iloc[0]["top_1_match_rate"] == pytest.approx(
        expected.top_1_match_rate
    )
    assert games.iloc[0]["centipawn_loss_stability"] == pytest.approx(
        expected.centipawn_loss_stability
    )
    assert game_feature_values(game_from_record(games.iloc[0].to_dict())) == pytest.approx(
        game_feature_values(expected)
    )


def test_feature_contract_has_no_leakage_and_rejects_forbidden_names() -> None:
    validate_game_feature_names(GAME_FEATURE_NAMES)
    assert_game_feature_contract(GAME_FEATURE_NAMES)
    assert not set(GAME_FEATURE_NAMES) & FORBIDDEN_COLUMNS

    with pytest.raises(FeatureContractError):
        validate_game_feature_names([*GAME_FEATURE_NAMES, "player_elo"])
    with pytest.raises(FeatureContractError):
        validate_game_feature_names([*GAME_FEATURE_NAMES, "is_cheating_player_game"])
    with pytest.raises(FeatureContractError):
        # Raw Elo smuggled in under a derived-looking name.
        validate_game_feature_names(["mean_centipawn_loss_player_elo"])
    with pytest.raises(FeatureContractError):
        assert_game_feature_contract(GAME_FEATURE_NAMES[:-1])


def test_missing_timing_sets_zero_and_has_timing_flag() -> None:
    rows = [
        _row("g1", 1, 21, thinking=float("nan")),
        _row("g1", 1, 22, thinking=float("nan")),
    ]
    games = build_game_frame(_frame(rows))
    record = games.iloc[0]
    assert record["median_time_seconds"] is None or math.isnan(
        record["median_time_seconds"]
    )
    assert record["has_timing"] == 0.0

    game = game_from_record(record.to_dict())
    vector = game_feature_vector(game)
    values = dict(zip(GAME_FEATURE_NAMES, vector))
    assert values["has_timing"] == 0.0
    assert values["median_time_seconds"] == 0.0
    assert values["mean_time_seconds"] == 0.0
    assert values["time_variance"] == 0.0
    assert all(math.isfinite(value) for value in vector)


def test_feature_vector_order_and_length() -> None:
    rows = [_row("g1", 1, 21, thinking=2.0), _row("g1", 1, 22, thinking=4.0)]
    games = build_game_frame(_frame(rows))
    game = game_from_record(games.iloc[0].to_dict())

    vector = game_feature_vector(game)
    values = game_feature_values(game)

    assert len(vector) == len(GAME_FEATURE_NAMES) == 11
    assert vector == [values[name] for name in GAME_FEATURE_NAMES]
    assert vector[GAME_FEATURE_NAMES.index("top_1_match_rate")] == pytest.approx(1.0)
    assert vector[GAME_FEATURE_NAMES.index("has_timing")] == 1.0


def test_build_game_dataset_returns_labels_groups_and_metadata() -> None:
    rows = [
        _row("g1", 1, 21, label=True),
        _row("g1", 1, 22, label=True),
        _row("g2", 2, 21, label=False),
        _row("g2", 2, 22, label=False),
    ]
    features, labels, groups, meta = build_game_dataset(_frame(rows))

    assert len(features) == len(labels) == len(groups) == len(meta) == 2
    assert labels == [1, 0]
    assert groups == [1, 2]
    assert list(meta.columns) == list(GAME_METADATA_COLUMNS)
    assert all(len(vector) == len(GAME_FEATURE_NAMES) for vector in features)
    assert not set(GAME_FEATURE_NAMES) & set(meta.columns)


def test_inconsistent_labels_and_multiple_players_are_rejected() -> None:
    mixed_label = [
        _row("g1", 1, 21, label=True),
        _row("g1", 1, 22, label=False),
    ]
    with pytest.raises(GameFeatureError, match="inconsistent"):
        build_game_frame(_frame(mixed_label))

    mixed_player = [
        _row("g1", 1, 21),
        _row("g1", 2, 22),
    ]
    with pytest.raises(GameFeatureError, match="multiple player_ids"):
        build_game_frame(_frame(mixed_player))
