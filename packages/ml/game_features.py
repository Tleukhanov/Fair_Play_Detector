"""Game-level ChessFraud features shared by training and serving.

ChessFraud labels decision points (moves), but the serving contract scores a
*player* from a set of analysed games.  This module is the single bridge between
the two granularities:

* :func:`build_game_frame` turns raw ``chessfraud_full.parquet`` rows into one
  :class:`~packages.feature_engineering.schema.GameAggregateFeatures` row per
  player-game, reusing :func:`packages.chess_analysis.features.aggregate_game_features`.
  Exactly the same aggregation runs on serve-time analysis results, so a trained
  game-level model sees the same definitions in both paths.
* :data:`GAME_FEATURE_NAMES` is the persisted model input contract.  It is a
  strict subset of the aggregate fields: only signals that are (a) populated in
  ChessFraud and (b) populated at serve time are allowed.  ``top_3_match_rate``
  is constant zero in ChessFraud, ``difficult_position_match_rate`` duplicates
  ``top_1_match_rate``, ``maia_divergence``/``time_difficulty_correlation`` are
  never populated, and ``move_count`` is a length confound.

Missing timing is explicit rather than imputed with a fake value: the three time
features become ``0.0`` and ``has_timing`` becomes ``0.0``.  ``has_timing``
lets the model separate "no clocks in the PGN" from "very fast moves".

Leakage policy is inherited from :mod:`packages.ml.feature_contract`: no label,
identity, raw Elo, raw eval, or raw clock column can appear in the model matrix.
:func:`validate_game_feature_names` fails closed on the blacklist and on names
that hint at raw Elo or labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from typing import TYPE_CHECKING, Any

from packages.chess_analysis.features import aggregate_game_features
from packages.chess_analysis.models import EVAL_CLIP_CP, MoveAnalysis
from packages.feature_engineering.schema import GameAggregateFeatures
from packages.ml.feature_contract import (
    MIN_HALF_MOVE,
    FeatureContractError,
    assert_no_forbidden_columns,
)

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

#: Version of the game-level feature contract.  Bump when names/definitions move.
GAME_FEATURE_SCHEMA_VERSION = "1.0.0"

#: Persisted model input order.  Position matters: artifacts are column-indexed.
GAME_FEATURE_NAMES: tuple[str, ...] = (
    "mean_centipawn_loss",
    "median_centipawn_loss",
    "p90_centipawn_loss",
    "top_1_match_rate",
    "tactical_error_count",
    "centipawn_loss_stability",
    "identical_time_share",
    "median_time_seconds",
    "mean_time_seconds",
    "time_variance",
    "has_timing",
)

#: Non-feature columns that travel with each dataset row.
GAME_METADATA_COLUMNS: tuple[str, ...] = ("game_id", "player_id", "label")

#: Raw parquet columns required to build the aggregates and the label.
REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "is_used",
    "half_move",
    "position_before",
    "move_player",
    "move_thinking_time",
    "eval_before",
    "eval_after",
    "centipawn_loss",
    "is_cheating_player_game",
    "move_stockfish_1",
    "move_stockfish_9",
    "move_stockfish_15",
)

#: Analysis budget the ChessFraud tables were produced with.
ANALYSIS_DEPTH = 15

#: Tokens that must never appear in a game-level feature name.  They would
#: smuggle raw Elo, identity, or a label into the matrix under a new name.
_FORBIDDEN_NAME_TOKENS: tuple[str, ...] = (
    "elo",
    "label",
    "cheat",
    "player_id",
    "game_id",
    "opponent",
    "tournament",
    "rating",
    "result",
    "half_move",
    "move_player",
    "position",
)

_TIME_FEATURE_NAMES: tuple[str, ...] = (
    "median_time_seconds",
    "mean_time_seconds",
    "time_variance",
)


class GameFeatureError(ValueError):
    """Raised when a game-level feature row cannot be built honestly."""


def validate_game_feature_names(names: Iterable[str]) -> None:
    """Fail closed unless ``names`` are allowed in the game-level matrix.

    Checks for duplicates, the shared per-move blacklist, and tokens that would
    indicate raw Elo/label/identity leakage.  It intentionally does not require
    an exact column set; :func:`assert_game_feature_contract` does that.
    """

    actual = tuple(names)
    if len(actual) != len(set(actual)):
        duplicates = sorted({name for name in actual if actual.count(name) > 1})
        raise FeatureContractError(f"duplicate game feature columns: {duplicates}")
    try:
        assert_no_forbidden_columns(actual)
    except FeatureContractError as error:
        raise FeatureContractError(f"game feature contract violation: {error}") from error
    suspicious = sorted(
        name
        for name in actual
        if any(token in name.casefold() for token in _FORBIDDEN_NAME_TOKENS)
    )
    if suspicious:
        raise FeatureContractError(
            f"game feature names suggest raw labels/identity/Elo: {suspicious}"
        )


def assert_game_feature_contract(names: Iterable[str]) -> None:
    """Require exactly :data:`GAME_FEATURE_NAMES` in the persisted order."""

    actual = tuple(names)
    validate_game_feature_names(actual)
    if actual != GAME_FEATURE_NAMES:
        missing = [name for name in GAME_FEATURE_NAMES if name not in actual]
        unexpected = [name for name in actual if name not in GAME_FEATURE_NAMES]
        raise FeatureContractError(
            "game feature columns do not match the persisted contract; "
            f"missing={missing}, unexpected={unexpected}, order={actual}"
        )


def _optional_float(value: object, *, field: str) -> float | None:
    """Return a finite float or ``None`` for a missing/NaN value."""

    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise GameFeatureError(f"column {field!r} is not numeric: {value!r}") from error
    if not isfinite(number):
        return None
    return number


def _required_eval(value: object, *, field: str) -> int:
    number = _optional_float(value, field=field)
    if number is None:
        raise GameFeatureError(f"column {field!r} is required but missing")
    return int(round(max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, number))))


def _required_int(value: object, *, field: str) -> int:
    number = _optional_float(value, field=field)
    if number is None:
        raise GameFeatureError(f"column {field!r} is required but missing")
    return int(round(number))


def _matches(played: str, value: object) -> bool:
    return isinstance(value, str) and value == played


def move_analysis_from_row(row: Mapping[str, object]) -> MoveAnalysis:
    """Convert one ChessFraud parquet row into a focal-player ``MoveAnalysis``.

    ``eval_best_move_cp`` has no separate engine best-move eval in ChessFraud, so
    it mirrors ``eval_before``.  ``centipawn_loss`` is clipped to the model bound
    (the raw table contains a handful of values above 1000) and an explicit
    value is preserved so the aggregate uses it instead of recomputing.
    """

    played = str(row["move_player"])
    loss = _optional_float(row.get("centipawn_loss"), field="centipawn_loss")
    thinking = _optional_float(row.get("move_thinking_time"), field="move_thinking_time")
    top_1_sf1 = _matches(played, row.get("move_stockfish_1"))
    top_1_sf9 = _matches(played, row.get("move_stockfish_9"))
    top_1_sf15 = _matches(played, row.get("move_stockfish_15"))
    return MoveAnalysis(
        ply=_required_int(row.get("half_move"), field="half_move"),
        fen_before=str(row["position_before"]),
        played_move_uci=played,
        played_move_san=played,
        eval_before_cp=_required_eval(row.get("eval_before"), field="eval_before"),
        eval_best_move_cp=_required_eval(row.get("eval_before"), field="eval_before"),
        eval_played_move_cp=_required_eval(row.get("eval_after"), field="eval_after"),
        centipawn_loss=(
            None if loss is None else max(0.0, min(float(EVAL_CLIP_CP), loss))
        ),
        is_top_1=top_1_sf15,
        is_top_3=False,
        is_top1_sf1=top_1_sf1,
        is_top1_sf9=top_1_sf9,
        is_top1_sf15=top_1_sf15,
        time_seconds=None if thinking is None else max(0.0, thinking),
        analysis_depth=ANALYSIS_DEPTH,
        is_forced=False,
        is_noisy_position=False,
    )


def _select_used(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise GameFeatureError(f"raw frame is missing required columns: {missing}")
    selected = frame.loc[
        frame["is_used"].eq(True) & (frame["half_move"] >= MIN_HALF_MOVE)
    ]
    if selected.empty:
        raise GameFeatureError("no is_used rows with half_move >= 21 remain")
    return selected.sort_values(["game_id", "half_move"], kind="mergesort")


def _aggregate_group(rows: Sequence[Mapping[str, object]], game_id: str) -> dict[str, Any]:
    player_ids = {row["player_id"] for row in rows}
    if len(player_ids) != 1:
        raise GameFeatureError(
            f"game {game_id!r} spans multiple player_ids: {sorted(map(str, player_ids))}"
        )
    labels = {bool(row["is_cheating_player_game"]) for row in rows}
    if len(labels) != 1:
        raise GameFeatureError(f"game {game_id!r} has inconsistent player labels")

    moves = [move_analysis_from_row(row) for row in rows]
    aggregate = aggregate_game_features(moves)
    record: dict[str, Any] = {
        "game_id": game_id,
        "player_id": rows[0]["player_id"],
        "label": int(labels.pop()),
    }
    record.update(aggregate.model_dump())
    return record


def build_game_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player-game with aggregates and the game-level label.

    Rows are filtered to ``is_used == True`` and ``half_move >= 21``, then
    aggregated per ``game_id`` with :func:`aggregate_game_features`.  The result
    carries :data:`GAME_METADATA_COLUMNS` plus every
    :class:`GameAggregateFeatures` field; only :data:`GAME_FEATURE_NAMES` is
    passed to the model by :func:`build_game_dataset`.
    """

    import pandas as pd

    selected = _select_used(frame)
    records: list[dict[str, Any]] = []
    columns = [*GAME_METADATA_COLUMNS, *GameAggregateFeatures.model_fields, "has_timing"]
    for game_id, group in selected.groupby("game_id", sort=True):
        rows = group.to_dict(orient="records")
        record = _aggregate_group(rows, str(game_id))
        record["has_timing"] = 1.0 if record["median_time_seconds"] is not None else 0.0
        records.append({column: record[column] for column in columns})
    return pd.DataFrame.from_records(records, columns=columns)


def game_from_record(record: Mapping[str, object]) -> GameAggregateFeatures:
    """Validate a stored game row, mapping ``NaN`` to ``None`` for timing.

    Pandas turns ``None`` timing into ``NaN`` inside a float column, but the
    pydantic aggregate requires ``None`` (not ``NaN``) for absent timing.  Serve
    time already produces ``None``; this helper keeps training/serving consumers
    of stored frames aligned.
    """

    payload: dict[str, object] = {}
    for name in GameAggregateFeatures.model_fields:
        value = record[name]
        if isinstance(value, float) and not isfinite(value):
            value = None
        payload[name] = value
    return GameAggregateFeatures.model_validate(payload)


def game_feature_values(game: GameAggregateFeatures) -> dict[str, float]:
    """Map one aggregate to the persisted feature names and values."""

    values: dict[str, float] = {
        "mean_centipawn_loss": float(game.mean_centipawn_loss),
        "median_centipawn_loss": float(game.median_centipawn_loss),
        "p90_centipawn_loss": float(game.p90_centipawn_loss),
        "top_1_match_rate": float(game.top_1_match_rate),
        "tactical_error_count": float(game.tactical_error_count),
        "centipawn_loss_stability": float(game.centipawn_loss_stability),
        "identical_time_share": float(game.identical_time_share),
        "has_timing": 1.0 if game.median_time_seconds is not None else 0.0,
    }
    for name in _TIME_FEATURE_NAMES:
        raw = getattr(game, name)
        values[name] = 0.0 if raw is None else float(raw)
    for name, value in values.items():
        if not isfinite(value):
            raise GameFeatureError(f"game feature {name!r} is not finite: {value!r}")
    return values


def game_feature_vector(game: GameAggregateFeatures) -> list[float]:
    """Return the model input vector in :data:`GAME_FEATURE_NAMES` order."""

    values = game_feature_values(game)
    return [values[name] for name in GAME_FEATURE_NAMES]


def game_dataset_from_frame(
    games: pd.DataFrame,
) -> tuple[list[list[float]], list[int], list[object], pd.DataFrame]:
    """Build ``(X, y, groups, meta)`` from a frame returned by :func:`build_game_frame`.

    ``X`` follows :data:`GAME_FEATURE_NAMES`; ``groups`` are ``player_id`` values
    for player-disjoint splitting; ``meta`` contains ``game_id``, ``player_id``
    and ``label`` in the same row order.
    """

    records = games.loc[:, list(GameAggregateFeatures.model_fields)].to_dict(
        orient="records"
    )
    vectors = [game_feature_vector(game_from_record(record)) for record in records]
    labels = [int(value) for value in games["label"]]
    groups: list[object] = [value for value in games["player_id"]]
    meta = games.loc[:, list(GAME_METADATA_COLUMNS)].reset_index(drop=True)
    return vectors, labels, groups, meta


def build_game_dataset(
    frame: pd.DataFrame,
) -> tuple[list[list[float]], list[int], list[object], pd.DataFrame]:
    """Build ``(X, y, groups, meta)`` from a raw ChessFraud frame."""

    return game_dataset_from_frame(build_game_frame(frame))


__all__: Sequence[str] = (
    "ANALYSIS_DEPTH",
    "GAME_FEATURE_NAMES",
    "GAME_FEATURE_SCHEMA_VERSION",
    "GAME_METADATA_COLUMNS",
    "REQUIRED_SOURCE_COLUMNS",
    "GameFeatureError",
    "assert_game_feature_contract",
    "build_game_dataset",
    "build_game_frame",
    "game_dataset_from_frame",
    "game_feature_values",
    "game_from_record",
    "game_feature_vector",
    "move_analysis_from_row",
    "validate_game_feature_names",
)
