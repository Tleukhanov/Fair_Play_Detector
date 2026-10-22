"""Canonical model feature contract shared by training and inference.

The contract is deliberately dependency-free: the API process does not install
the optional ML dependencies, but it still needs the feature names and the
leakage blacklist.  Any change here is a schema change and must bump
``FEATURE_SCHEMA_VERSION``.

The contract answers one question: *which columns may a model see?*  Everything
that encodes the label, the player identity, the data source, or a raw value
that is only allowed through a derived/normalized feature is listed in
``FORBIDDEN_COLUMNS``.  ``validate_feature_columns`` fails closed, so a model
cannot be trained (or served) on a frame that violates the contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import IntEnum
from math import isfinite

FEATURE_SCHEMA_VERSION = "2.0.0"

TARGET_COLUMN = "is_cheating_move"
PLAYER_GAME_TARGET_COLUMN = "is_cheating_player_game"
GROUP_COLUMN = "player_id"

# ChessFraud labels moves after the first 10 full moves.  We keep the same
# opening cutoff so training and serving see the same positions.
OPENING_CUTOFF_HALF_MOVE = 20
MIN_HALF_MOVE = OPENING_CUTOFF_HALF_MOVE + 1


class Phase(IntEnum):
    """Coarse game phase; stored as a small integer feature."""

    OPENING = 0
    MIDDLEGAME = 1
    ENDGAME = 2


# The order is part of the artifact contract: persisted models index columns by
# position, not by name.
FEATURE_NAMES: tuple[str, ...] = (
    "centipawn_loss",
    "normalized_centipawn_loss",
    "match_sf1",
    "match_sf9",
    "match_sf15",
    "eval_swing",
    "abs_eval",
    "phase",
    "move_thinking_time",
    "time_ratio",
    "player_elo_centered",
    "opponent_elo_centered",
)

# Columns that must never appear in the final model matrix.  They either encode
# the label, the player/game identity, the data source, or a raw value that is
# only allowed through a derived feature (raw Elo, raw evals, raw clocks).
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        # selection / provenance
        "is_used",
        "game_id",
        "player_id",
        "opponent_id",
        "tournament_id",
        "rating_bin",
        "player_color",
        "time_control",
        "game_result",
        # labels and label-derived fields
        "player_hint_shown",
        "assistance_search_depth",
        "assistance_line_rank",
        "is_cheating_move",
        "is_cheating_player_game",
        "is_accused_by_opponent",
        # raw board/move fields; used to derive match_* and CPL only
        "move_player",
        "position_before",
        "position_after",
        "move_stockfish_1",
        "move_stockfish_5",
        "move_stockfish_9",
        "move_stockfish_11",
        "move_stockfish_15",
        "move_lc0_1",
        "move_lc0_10",
        "move_lc0_100",
        "move_maia1_1900",
        "move_maia2_2050",
        "move_allie_2500",
        "maia2_win_prob_2050",
        "allie_win_prob_2500",
        # raw values replaced by normalized/derived features
        "player_elo",
        "opponent_elo",
        "eval_before",
        "eval_after",
        "clock_remaining_time",
        "half_move",
    }
)


class FeatureContractError(ValueError):
    """Raised when a feature frame violates the shared contract."""


def validate_feature_columns(columns: Iterable[str]) -> None:
    """Fail closed unless ``columns`` is exactly the agreed feature vector."""

    actual = tuple(columns)
    if len(actual) != len(set(actual)):
        duplicates = sorted({name for name in actual if actual.count(name) > 1})
        raise FeatureContractError(f"duplicate feature columns: {duplicates}")
    missing = [name for name in FEATURE_NAMES if name not in actual]
    if missing:
        raise FeatureContractError(f"missing feature columns: {missing}")
    unexpected = [name for name in actual if name not in FEATURE_NAMES]
    if unexpected:
        raise FeatureContractError(f"unexpected feature columns: {unexpected}")
    if actual != FEATURE_NAMES:
        raise FeatureContractError(
            "feature column order does not match the persisted contract"
        )
    assert_no_forbidden_columns(actual)


def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    """Raise if any label/leakage column is present in ``columns``."""

    forbidden = sorted(set(columns) & FORBIDDEN_COLUMNS)
    if forbidden:
        raise FeatureContractError(f"forbidden leakage columns present: {forbidden}")


def validate_player_disjoint(
    train_players: Iterable[object],
    test_players: Iterable[object],
    *,
    split_name: str = "split",
) -> None:
    """Raise if a player appears in both sides of a split."""

    overlap = set(train_players) & set(test_players)
    if overlap:
        sample = sorted(str(player) for player in overlap)[:10]
        raise FeatureContractError(
            f"{split_name} is not player-disjoint; overlapping players: {sample}"
        )


def validate_feature_values(rows: Iterable[Mapping[str, object]]) -> None:
    """Validate a materialized feature matrix without depending on pandas.

    Checks that every row has exactly the contract columns, every value is a
    finite number, and ``phase`` is one of the three known values.
    """

    for index, row in enumerate(rows):
        if set(row) != set(FEATURE_NAMES):
            missing = sorted(set(FEATURE_NAMES) - set(row))
            unexpected = sorted(set(row) - set(FEATURE_NAMES))
            raise FeatureContractError(
                f"row {index} does not match the feature contract; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name in FEATURE_NAMES:
            value = row[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise FeatureContractError(
                    f"row {index} feature {name!r} is not numeric: {type(value).__name__}"
                )
            if not isfinite(float(value)):
                raise FeatureContractError(
                    f"row {index} feature {name!r} is not finite: {value!r}"
                )
        phase_value = row["phase"]
        if not isinstance(phase_value, (int, float)) or isinstance(phase_value, bool):
            raise FeatureContractError(
                f"row {index} has non-numeric phase value: {phase_value!r}"
            )
        if int(phase_value) not in tuple(int(phase) for phase in Phase):
            raise FeatureContractError(
                f"row {index} has unknown phase value: {phase_value!r}"
            )


def feature_vector(row: Mapping[str, object]) -> Sequence[float]:
    """Return the model input vector in the persisted column order."""

    values: list[float] = []
    for name in FEATURE_NAMES:
        value = row[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise FeatureContractError(
                f"feature {name!r} is not numeric: {type(value).__name__}"
            )
        values.append(float(value))
    return tuple(values)
