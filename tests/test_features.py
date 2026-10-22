"""Unit tests for the canonical ChessFraud training/serve feature derivation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.ml.feature_contract import (
    FEATURE_NAMES,
    FORBIDDEN_COLUMNS,
    FeatureContractError,
    validate_feature_columns,
    validate_feature_values,
)
from packages.ml.features import (
    METADATA_COLUMNS,
    ServeMove,
    build_serve_features,
    build_training_features,
    build_training_metadata,
    fit_elo_centering,
    normalized_centipawn_loss_from_evals,
    parse_time_control_base,
)

# Hand-written train stats: the two rating bins of the fixture plus a global
# fallback, exactly like fit_elo_centering would persist them.
CENTERING: dict[str, float] = {
    "player_elo|1600-1800": 1600.0,
    "player_elo|2000-2200": 2000.0,
    "player_elo|__global__": 1800.0,
    "opponent_elo|1600-1800": 1750.0,
    "opponent_elo|2000-2200": 2100.0,
    "opponent_elo|__global__": 1900.0,
}


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "g1",
        "player_id": "p1",
        "is_used": True,
        "half_move": 30,
        "player_elo": 1650.0,
        "opponent_elo": 1700.0,
        "time_control": "180+0",
        "rating_bin": "1600-1800",
        "move_player": "e2e4",
        "move_stockfish_1": "d2d4",
        "move_stockfish_9": "g1f3",
        "move_stockfish_15": "c2c4",
        "eval_before": 40.0,
        "eval_after": 20.0,
        "centipawn_loss": 20.0,
        "move_thinking_time": 12.0,
        "clock_remaining_time": 90.0,
        "is_cheating_move": False,
        "is_cheating_player_game": False,
    }
    row.update(overrides)
    row["normalized_centipawn_loss"] = normalized_centipawn_loss_from_evals(
        float(row["eval_before"]), float(row["eval_after"])
    )
    return row


def _training_frame() -> pd.DataFrame:
    human = _row()
    assisted = _row(
        game_id="g2",
        player_id="p2",
        half_move=70,
        player_elo=2050.0,
        opponent_elo=2100.0,
        time_control="300+2",
        rating_bin="2000-2200",
        move_player="d2d4",
        move_stockfish_1="d2d4",
        move_stockfish_9="d2d4",
        move_stockfish_15="g1f3",
        eval_before=-100.0,
        eval_after=-300.0,
        centipawn_loss=200.0,
        move_thinking_time=480.0,
        clock_remaining_time=600.0,
        is_cheating_move=True,
        is_cheating_player_game=True,
    )
    opening = _row(
        game_id="g3",
        player_id="p3",
        half_move=10,
        player_elo=1000.0,
        opponent_elo=1100.0,
        rating_bin="lt_1200",
    )
    unused = _row(
        game_id="g4",
        player_id="p4",
        is_used=False,
        player_elo=1050.0,
        opponent_elo=1150.0,
        rating_bin="lt_1200",
    )
    return pd.DataFrame([human, assisted, opening, unused])


def test_features_are_exactly_the_contract_in_order() -> None:
    features = build_training_features(_training_frame(), elo_centering=CENTERING)

    assert list(features.columns) == list(FEATURE_NAMES)
    assert len(features) == 2  # opening and unused rows are gated out
    validate_feature_columns(list(features.columns))
    validate_feature_values(features.to_dict("records"))


def test_no_forbidden_columns_leak_into_the_matrix() -> None:
    features = build_training_features(_training_frame(), elo_centering=CENTERING)

    assert not set(features.columns) & FORBIDDEN_COLUMNS
    for name in ("player_id", "game_id", "half_move", "is_cheating_move"):
        assert name not in features.columns


def test_known_values_for_human_and_assisted_rows() -> None:
    features = build_training_features(_training_frame(), elo_centering=CENTERING)
    human = features.iloc[0]
    assisted = features.iloc[1]

    # Human-like move: off-book at every checked depth, small loss, early phase.
    assert human["centipawn_loss"] == pytest.approx(20.0)
    assert human["normalized_centipawn_loss"] == pytest.approx(
        normalized_centipawn_loss_from_evals(40.0, 20.0)
    )
    assert human["match_sf1"] == pytest.approx(0.0)
    assert human["match_sf9"] == pytest.approx(0.0)
    assert human["match_sf15"] == pytest.approx(0.0)
    assert human["eval_swing"] == pytest.approx(-20.0)
    assert human["abs_eval"] == pytest.approx(40.0)
    assert human["phase"] == 1
    assert human["move_thinking_time"] == pytest.approx(12.0)
    assert human["time_ratio"] == pytest.approx(90.0 / 180.0)

    # Assisted-like move: engine top-1 at depth 1 and 9, endgame, clocks clipped.
    assert assisted["match_sf1"] == pytest.approx(1.0)
    assert assisted["match_sf9"] == pytest.approx(1.0)
    assert assisted["match_sf15"] == pytest.approx(0.0)
    assert assisted["centipawn_loss"] == pytest.approx(200.0)
    assert assisted["eval_swing"] == pytest.approx(-200.0)
    assert assisted["abs_eval"] == pytest.approx(100.0)
    assert assisted["phase"] == 2
    assert assisted["move_thinking_time"] == pytest.approx(300.0)  # clipped to base
    assert assisted["time_ratio"] == pytest.approx(1.5)  # 600 / 300, clipped


def test_values_are_clipped_to_contract_bounds() -> None:
    frame = _training_frame()
    frame.loc[0, "centipawn_loss"] = 5000.0
    frame.loc[0, "eval_before"] = -5000.0
    frame.loc[0, "eval_after"] = 5000.0
    features = build_training_features(frame, elo_centering=CENTERING)

    assert features.loc[0, "centipawn_loss"] == pytest.approx(1000.0)
    assert features.loc[0, "eval_swing"] == pytest.approx(1000.0)
    assert features.loc[0, "abs_eval"] == pytest.approx(1000.0)


def test_metadata_stays_outside_the_matrix_and_stays_aligned() -> None:
    frame = _training_frame()
    features = build_training_features(frame, elo_centering=CENTERING)
    metadata = build_training_metadata(frame)

    assert list(metadata.columns) == list(METADATA_COLUMNS)
    assert len(metadata) == len(features)
    assert metadata["player_id"].tolist() == ["p1", "p2"]
    assert metadata["game_id"].tolist() == ["g1", "g2"]
    assert metadata["is_cheating_move"].tolist() == [False, True]


def test_metadata_without_label_columns_matches_synth_schema() -> None:
    frame = _training_frame().drop(columns=["is_cheating_move", "is_cheating_player_game"])
    features = build_training_features(frame, elo_centering=CENTERING)
    metadata = build_training_metadata(frame)

    assert list(metadata.columns) == ["player_id", "game_id", "half_move"]
    assert len(metadata) == len(features)


def test_fit_elo_centering_uses_only_eligible_training_rows() -> None:
    centering = fit_elo_centering(_training_frame())

    assert centering["player_elo|1600-1800"] == pytest.approx(1650.0)
    assert centering["opponent_elo|1600-1800"] == pytest.approx(1700.0)
    assert centering["player_elo|2000-2200"] == pytest.approx(2050.0)
    assert centering["opponent_elo|2000-2200"] == pytest.approx(2100.0)
    # Excluded opening/unused rows (1000/1050) must not move the global median.
    assert centering["player_elo|__global__"] == pytest.approx(1850.0)
    assert centering["opponent_elo|__global__"] == pytest.approx(1900.0)


def test_provided_train_centering_is_applied_verbatim() -> None:
    features = build_training_features(_training_frame(), elo_centering=CENTERING)

    assert features.loc[0, "player_elo_centered"] == pytest.approx(1650.0 - 1600.0)
    assert features.loc[0, "opponent_elo_centered"] == pytest.approx(1700.0 - 1750.0)
    assert features.loc[1, "player_elo_centered"] == pytest.approx(2050.0 - 2000.0)
    assert features.loc[1, "opponent_elo_centered"] == pytest.approx(2100.0 - 2100.0)


def test_default_centering_is_fit_on_the_passed_frame() -> None:
    features = build_training_features(_training_frame())
    centering = features.attrs["elo_centering"]

    assert centering["player_elo|1600-1800"] == pytest.approx(1650.0)
    assert features.loc[0, "player_elo_centered"] == pytest.approx(0.0)
    assert features.loc[0, "opponent_elo_centered"] == pytest.approx(0.0)


def test_unseen_rating_bin_falls_back_to_global_median() -> None:
    features = build_serve_features(
        [
            ServeMove(
                ply=30,
                eval_before_cp=10.0,
                eval_played_cp=0.0,
                is_top1_sf1=True,
                is_top1_sf9=False,
                is_top1_sf15=False,
            )
        ],
        player_elo=2500.0,
        opponent_elo=2400.0,
        time_control="180+0",
        elo_centering=CENTERING,
    )

    assert features.loc[0, "player_elo_centered"] == pytest.approx(2500.0 - 1800.0)
    assert features.loc[0, "opponent_elo_centered"] == pytest.approx(2400.0 - 1900.0)


def test_fail_closed_on_missing_source_column() -> None:
    frame = _training_frame().drop(columns=["eval_before"])
    with pytest.raises(FeatureContractError, match="missing required source columns"):
        build_training_features(frame, elo_centering=CENTERING)


def test_fail_closed_on_nan_in_a_used_row() -> None:
    frame = _training_frame()
    frame.loc[0, "eval_before"] = float("nan")
    with pytest.raises(FeatureContractError, match="missing values"):
        build_training_features(frame, elo_centering=CENTERING)

    frame = _training_frame()
    frame.loc[0, "move_player"] = None
    with pytest.raises(FeatureContractError, match="missing values"):
        build_training_features(frame, elo_centering=CENTERING)


def test_missing_timing_fails_closed_unless_explicitly_allowed() -> None:
    frame = _training_frame()
    frame.loc[0, "move_thinking_time"] = float("nan")

    with pytest.raises(FeatureContractError, match="move_thinking_time"):
        build_training_features(frame, elo_centering=CENTERING)

    features = build_training_features(
        frame, elo_centering=CENTERING, allow_missing_timing=True
    )
    assert features.loc[0, "move_thinking_time"] == pytest.approx(0.0)


def test_fail_closed_on_nan_half_move() -> None:
    frame = _training_frame()
    frame["half_move"] = frame["half_move"].astype(float)
    frame.loc[0, "half_move"] = float("nan")
    with pytest.raises(FeatureContractError, match="half_move"):
        build_training_features(frame, elo_centering=CENTERING)


def test_fail_closed_on_forbidden_column_in_a_feature_matrix() -> None:
    features = build_training_features(_training_frame(), elo_centering=CENTERING)
    leaked_columns = [*features.columns, "player_id"]
    with pytest.raises(FeatureContractError, match="forbidden|unexpected"):
        validate_feature_columns(leaked_columns)

    row = features.iloc[0].to_dict()
    row["is_cheating_move"] = 0.0
    with pytest.raises(FeatureContractError, match="does not match"):
        validate_feature_values([row])


@pytest.mark.parametrize(
    ("control", "expected"), [("180+0", 180.0), ("300+2", 300.0), ("60+1", 60.0)]
)
def test_time_control_parsing(control: str, expected: float) -> None:
    assert parse_time_control_base(control) == expected


@pytest.mark.parametrize("control", ["blitz", "180", "180+", "0+0", "-60+0", "180+x", ""])
def test_time_control_parsing_rejects_garbage(control: str) -> None:
    with pytest.raises(FeatureContractError):
        parse_time_control_base(control)


def test_normalized_cpl_formula_is_bounded_and_one_sided() -> None:
    gaining = normalized_centipawn_loss_from_evals(0.0, 100.0)
    losing = normalized_centipawn_loss_from_evals(100.0, 0.0)

    assert gaining == pytest.approx(0.0)
    assert losing == pytest.approx(1.0 / (1.0 + math.exp(-0.00368208 * 100.0)) - 0.5)
    assert 0.0 <= gaining <= 1.0
    assert 0.0 <= losing <= 1.0


def _serve_moves() -> list[ServeMove]:
    return [
        ServeMove(
            ply=30,
            eval_before_cp=40.0,
            eval_played_cp=20.0,
            is_top1_sf1=False,
            is_top1_sf9=False,
            is_top1_sf15=False,
            time_seconds=12.0,
            clock_seconds=90.0,
        ),
        ServeMove(
            ply=10,
            eval_before_cp=0.0,
            eval_played_cp=0.0,
            is_top1_sf1=True,
            is_top1_sf9=True,
            is_top1_sf15=True,
        ),
        ServeMove(
            ply=24,
            eval_before_cp=0.0,
            eval_played_cp=0.0,
            is_top1_sf1=True,
            is_top1_sf9=True,
            is_top1_sf15=True,
            is_forced=True,
        ),
    ]


def test_serve_features_match_the_training_row_that_saw_the_same_move() -> None:
    training = build_training_features(_training_frame(), elo_centering=CENTERING)
    serve = build_serve_features(
        _serve_moves()[:1],
        player_elo=1650.0,
        opponent_elo=1700.0,
        time_control="180+0",
        elo_centering=CENTERING,
    )

    pd.testing.assert_frame_equal(
        training.iloc[[0]].reset_index(drop=True), serve.reset_index(drop=True)
    )


def test_serve_features_are_deterministic() -> None:
    kwargs = {
        "player_elo": 1650.0,
        "opponent_elo": 1700.0,
        "time_control": "180+0",
        "elo_centering": CENTERING,
    }
    first = build_serve_features(_serve_moves(), **kwargs)
    second = build_serve_features(_serve_moves(), **kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(FEATURE_NAMES)


def test_serve_filters_forced_and_opening_moves() -> None:
    features = build_serve_features(
        _serve_moves(),
        player_elo=1650.0,
        opponent_elo=1700.0,
        time_control="180+0",
        elo_centering=CENTERING,
    )

    assert len(features) == 1
    assert features.loc[0, "centipawn_loss"] == pytest.approx(20.0)
    assert features.loc[0, "phase"] == 1


def test_serve_returns_empty_contract_frame_when_no_move_survives() -> None:
    features = build_serve_features(
        [move for move in _serve_moves() if move.is_forced or move.ply < 21],
        player_elo=1650.0,
        opponent_elo=1700.0,
        time_control="180+0",
        elo_centering=CENTERING,
    )

    assert features.empty
    assert list(features.columns) == list(FEATURE_NAMES)
    validate_feature_columns(list(features.columns))


def test_serve_requires_train_elo_centering() -> None:
    with pytest.raises(FeatureContractError, match="elo_centering"):
        build_serve_features(
            _serve_moves(),
            player_elo=1650.0,
            opponent_elo=1700.0,
            time_control="180+0",
        )


def test_serve_missing_clocks_degrade_to_zero() -> None:
    features = build_serve_features(
        [
            ServeMove(
                ply=30,
                eval_before_cp=10.0,
                eval_played_cp=0.0,
                is_top1_sf1=False,
                is_top1_sf9=False,
                is_top1_sf15=False,
            )
        ],
        player_elo=1650.0,
        opponent_elo=1700.0,
        time_control="300+2",
        elo_centering=CENTERING,
    )

    assert features.loc[0, "move_thinking_time"] == pytest.approx(0.0)
    assert features.loc[0, "time_ratio"] == pytest.approx(0.0)


def test_serve_time_ratio_uses_the_parsed_base() -> None:
    features = build_serve_features(
        [
            ServeMove(
                ply=30,
                eval_before_cp=0.0,
                eval_played_cp=0.0,
                is_top1_sf1=False,
                is_top1_sf9=False,
                is_top1_sf15=False,
                time_seconds=10.0,
                clock_seconds=30.0,
            )
        ],
        player_elo=1650.0,
        opponent_elo=1700.0,
        time_control="60+1",
        elo_centering=CENTERING,
    )

    assert features.loc[0, "time_ratio"] == pytest.approx(0.5)
    assert features.loc[0, "move_thinking_time"] == pytest.approx(10.0)


def test_serve_move_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="ply"):
        ServeMove(
            ply=0,
            eval_before_cp=0.0,
            eval_played_cp=0.0,
            is_top1_sf1=False,
            is_top1_sf9=False,
            is_top1_sf15=False,
        )
    with pytest.raises(ValueError, match="finite"):
        ServeMove(
            ply=30,
            eval_before_cp=float("nan"),
            eval_played_cp=0.0,
            is_top1_sf1=False,
            is_top1_sf9=False,
            is_top1_sf15=False,
        )
    with pytest.raises(ValueError, match="non-negative"):
        ServeMove(
            ply=30,
            eval_before_cp=0.0,
            eval_played_cp=0.0,
            is_top1_sf1=False,
            is_top1_sf9=False,
            is_top1_sf15=False,
            clock_seconds=-1.0,
        )
