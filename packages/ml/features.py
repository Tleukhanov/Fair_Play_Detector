"""Canonical ChessFraud move features, shared by training and serving.

This module is the single implementation of the v1 feature vector defined in
:mod:`packages.ml.feature_contract`.  Training reads ChessFraud Parquet rows,
inference reads analyzed moves (:class:`ServeMove`); both paths materialize the
same 12 columns through :func:`_feature_matrix`, so parity is structural rather
than a convention.

Key decisions
-------------
* Rows are gated before any arithmetic: ``is_used`` must be true and
  ``half_move >= MIN_HALF_MOVE`` (21), i.e. the opening is dropped exactly like
  ChessFraud's labelled subset.  For serving, ``ply < 21`` moves are skipped.
* Every raw column that is not a model feature stays outside the matrix.  The
  returned frame contains exactly :data:`FEATURE_NAMES`; labels and identifiers
  are available through :func:`build_training_metadata` (same row order).
* Elo is never a raw feature.  :func:`fit_elo_centering` returns medians per
  ``rating_bin`` (falling back to a global median for unseen bins) and the
  centered values are ``elo - median``.  **Fit on the training split only** and
  pass the same mapping to validation/test/serving.
* ``normalized_centipawn_loss`` for serving is recomputed from evals with the
  exact formula ChessFraud uses for its stored column:
  ``max(0, w(eval_before) - w(eval_after))`` where
  ``w(cp) = 1 / (1 + exp(-0.00368208 * cp))``.  The identity was verified
  against both ChessFraud Parquets (max abs error ~2e-16).
* ``time_control`` is parsed as ``base+increment`` in **seconds** (ChessFraud
  and Lichess both use seconds); the base is never hardcoded.

Missing data
------------
Training fails closed: a missing required column, a NaN in any required input,
or an unparsable ``time_control`` raises :class:`FeatureContractError`.  The
only exception is timing: ChessFraud synth contains a handful of eligible rows
with a NaN ``move_thinking_time``.  Those rows raise by default and can be
zero-filled with ``allow_missing_timing=True`` (the same degradation the serve
path applies when clocks are absent).  On serve, a missing thinking time becomes
0 seconds and a missing clock becomes ``time_ratio = 0.0``; both are documented
degradations, not silent imputations of an unavailable signal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from packages.ml.feature_contract import (
    FEATURE_NAMES,
    MIN_HALF_MOVE,
    FeatureContractError,
    Phase,
    validate_feature_columns,
    validate_feature_values,
)

# Clipping bounds are part of the feature definition, not a tuning knob.
MAX_CENTIPAWNS = 1000.0
MAX_TIME_RATIO = 1.5
ENDGAME_LAST_HALF_MOVE = 60
RATING_BIN_WIDTH = 200
WIN_PROBABILITY_SCALE = 0.00368208

ELO_CENTERING_SEPARATOR = "|"
GLOBAL_ELO_BIN = "__global__"

ELIGIBILITY_COLUMNS: tuple[str, ...] = ("is_used", "half_move")

REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "player_elo",
    "opponent_elo",
    "time_control",
    "move_player",
    "move_stockfish_1",
    "move_stockfish_9",
    "move_stockfish_15",
    "eval_before",
    "eval_after",
    "centipawn_loss",
    "normalized_centipawn_loss",
    "move_thinking_time",
    "clock_remaining_time",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "player_id",
    "game_id",
    "half_move",
    "is_cheating_move",
    "is_cheating_player_game",
)

_LABEL_COLUMNS: frozenset[str] = frozenset({"is_cheating_move", "is_cheating_player_game"})

_NUMERIC_SOURCE_COLUMNS: tuple[str, ...] = (
    "player_elo",
    "opponent_elo",
    "eval_before",
    "eval_after",
    "centipawn_loss",
    "normalized_centipawn_loss",
)

_TEXT_SOURCE_COLUMNS: tuple[str, ...] = (
    "time_control",
    "move_player",
    "move_stockfish_1",
    "move_stockfish_9",
    "move_stockfish_15",
)

_TIMING_SOURCE_COLUMNS: tuple[str, ...] = ("move_thinking_time", "clock_remaining_time")

_INTERMEDIATE_COLUMNS: tuple[str, ...] = (
    "centipawn_loss",
    "normalized_centipawn_loss",
    "match_sf1",
    "match_sf9",
    "match_sf15",
    "eval_before",
    "eval_after",
    "half_move",
    "move_thinking_time",
    "clock_remaining_time",
    "base_seconds",
    "player_elo",
    "opponent_elo",
    "rating_bin",
)

_VALIDATION_CHUNK_SIZE = 20_000


# ---------------------------------------------------------------------------
# Input DTO for the serve path
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServeMove:
    """One analyzed move from the requested player's point of view.

    The analysis layer (`packages/chess_analysis`) is responsible for converting
    evals to **mover POV** before constructing this object, exactly like the
    ChessFraud columns the model was trained on.

    ``clock_seconds`` is optional on purpose: live games may lack clocks.  When
    it is missing ``time_ratio`` is 0.0 (documented degradation), because there
    is no leak-free way to impute the remaining clock.
    """

    ply: int
    eval_before_cp: float
    eval_played_cp: float
    is_top1_sf1: bool
    is_top1_sf9: bool
    is_top1_sf15: bool
    time_seconds: float | None = None
    is_forced: bool = False
    clock_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ply, int) or isinstance(self.ply, bool) or self.ply < 1:
            raise ValueError(f"ply must be a positive integer, got {self.ply!r}")
        for name, value in (
            ("eval_before_cp", self.eval_before_cp),
            ("eval_played_cp", self.eval_played_cp),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        for name, optional_value in (
            ("time_seconds", self.time_seconds),
            ("clock_seconds", self.clock_seconds),
        ):
            if optional_value is None:
                continue
            if isinstance(optional_value, bool) or not isinstance(
                optional_value, (int, float)
            ):
                raise ValueError(f"{name} must be numeric or None, got {optional_value!r}")
            if not isfinite(float(optional_value)) or optional_value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {optional_value!r}")


# ---------------------------------------------------------------------------
# Time control parsing
# ---------------------------------------------------------------------------


def parse_time_control_base(time_control: object) -> float:
    """Return the base time in seconds from a ``base+increment`` control.

    ChessFraud and Lichess both express the base in seconds (``"180+0"`` is a
    3-minute game).  Anything else fails closed: guessing minutes vs seconds
    would silently corrupt ``time_ratio`` at serve time.
    """

    if not isinstance(time_control, str):
        raise FeatureContractError(
            f"time_control must be a string like 'base+increment', got {time_control!r}"
        )
    base_token, separator, increment_token = time_control.strip().partition("+")
    if not separator:
        raise FeatureContractError(
            f"time_control {time_control!r} is not in 'base+increment' seconds format"
        )
    try:
        base = float(base_token)
        increment = float(increment_token)
    except ValueError as error:
        raise FeatureContractError(
            f"time_control {time_control!r} has a non-numeric base or increment"
        ) from error
    if not isfinite(base) or base <= 0:
        raise FeatureContractError(f"time_control {time_control!r} has an invalid base: {base!r}")
    if not isfinite(increment) or increment < 0:
        raise FeatureContractError(
            f"time_control {time_control!r} has an invalid increment: {increment!r}"
        )
    return base


def normalized_centipawn_loss_from_evals(eval_before: float, eval_after: float) -> float:
    """Return ChessFraud's normalized CPL from mover-POV centipawn evals.

    This reproduces the stored ``normalized_centipawn_loss`` column exactly (see
    the module docstring), which is what makes train/serve parity possible.
    """

    return max(0.0, min(1.0, _win_probability(eval_before) - _win_probability(eval_after)))


def _win_probability(centipawns: float) -> float:
    return 1.0 / (1.0 + exp(-WIN_PROBABILITY_SCALE * centipawns))


# ---------------------------------------------------------------------------
# Elo normalization
# ---------------------------------------------------------------------------


def fit_elo_centering(df: pd.DataFrame) -> dict[str, float]:
    """Fit per-``rating_bin`` Elo medians on the training split.

    Returns a flat mapping keyed by ``"<player_elo|opponent_elo>|<bin>"`` plus a
    ``"<kind>|__global__"`` fallback used for rating bins that never appeared in
    training.  Rows are gated with ``is_used``/``half_move`` when those columns
    are present, so callers may pass either the raw Parquet frame or an already
    filtered frame.

    **Fit on train only.**  Passing validation/test rows here leaks cohort
    statistics into the model; persist this mapping in the model artifact and
    pass it to :func:`build_training_features` and :func:`build_serve_features`.
    """

    frame = _eligible_or_all(df)
    _require_columns(frame, ("player_elo", "opponent_elo"))
    if frame.empty:
        raise FeatureContractError("cannot fit Elo centering on an empty frame")
    return _fit_elo_centering_on_rows(
        frame,
        _to_float_column(frame, "player_elo"),
        _to_float_column(frame, "opponent_elo"),
    )


# ---------------------------------------------------------------------------
# Training features
# ---------------------------------------------------------------------------


def build_training_features(
    df: pd.DataFrame,
    *,
    elo_centering: Mapping[str, float] | None = None,
    allow_missing_timing: bool = False,
) -> pd.DataFrame:
    """Materialize the 12-column contract from ChessFraud Parquet rows.

    Only ``is_used`` rows with ``half_move >= 21`` are used.  The returned frame
    contains exactly :data:`FEATURE_NAMES` in contract order and nothing else;
    identifiers and labels are available via :func:`build_training_metadata`
    (same row order and length).

    Parameters
    ----------
    df:
        Raw ChessFraud (or synth) frame.
    elo_centering:
        Mapping from :func:`fit_elo_centering` **fitted on the training split**.
        When omitted it is fitted on ``df`` itself, which is only correct when
        ``df`` is the training split.  Validation/test callers must pass the
        train stats or the Elo normalization will not be comparable.
    allow_missing_timing:
        Zero-fill NaN ``move_thinking_time``/``clock_remaining_time`` instead of
        failing closed.  This mirrors the serve-path degradation and exists
        because synth contains a few eligible rows without clocks.

    The fitted centering mapping is attached as ``frame.attrs["elo_centering"]``
    so it can be persisted in the model artifact.
    """

    frame = _select_eligible(df)
    _require_columns(frame, (*REQUIRED_SOURCE_COLUMNS, "half_move"))
    _reject_missing_text(frame, _TEXT_SOURCE_COLUMNS)

    numeric = {column: _to_float_column(frame, column) for column in _NUMERIC_SOURCE_COLUMNS}
    half_move = _to_float_column(frame, "half_move")
    thinking = _to_float_column(frame, "move_thinking_time", allow_missing=allow_missing_timing)
    clock = _to_float_column(frame, "clock_remaining_time", allow_missing=allow_missing_timing)
    base_seconds = _time_control_bases(frame["time_control"])

    centering = (
        dict(elo_centering)
        if elo_centering is not None
        else _fit_elo_centering_on_rows(frame, numeric["player_elo"], numeric["opponent_elo"])
    )
    _validate_elo_centering(centering)

    intermediate = pd.DataFrame(
        {
            "centipawn_loss": numeric["centipawn_loss"].to_numpy(),
            "normalized_centipawn_loss": numeric["normalized_centipawn_loss"].to_numpy(),
            "match_sf1": (
                frame["move_player"] == frame["move_stockfish_1"]
            ).astype(float).to_numpy(),
            "match_sf9": (
                frame["move_player"] == frame["move_stockfish_9"]
            ).astype(float).to_numpy(),
            "match_sf15": (
                frame["move_player"] == frame["move_stockfish_15"]
            ).astype(float).to_numpy(),
            "eval_before": numeric["eval_before"].to_numpy(),
            "eval_after": numeric["eval_after"].to_numpy(),
            "half_move": half_move.to_numpy(),
            "move_thinking_time": thinking.to_numpy(),
            "clock_remaining_time": clock.to_numpy(),
            "base_seconds": base_seconds.to_numpy(),
            "player_elo": numeric["player_elo"].to_numpy(),
            "opponent_elo": numeric["opponent_elo"].to_numpy(),
            "rating_bin": _rating_bins(frame, numeric["player_elo"]).to_numpy(),
        },
        index=frame.index,
    )
    features = _feature_matrix(intermediate, centering)
    features.attrs["elo_centering"] = dict(centering)
    return features


def build_training_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return label/identity columns for the rows used by training features.

    Applies the same ``is_used``/``half_move`` gate and preserves input order,
    so ``build_training_metadata(df).reset_index(drop=True)`` aligns row by row
    with ``build_training_features(df)``.

    ``player_id``/``game_id``/``half_move`` are always required.  The label
    columns are optional because ``chess_fraud_synth`` does not ship them: the
    training pipeline constructs labels from the move sources before calling
    this helper.  Whatever label columns are present are passed through.
    """

    frame = _select_eligible(df)
    required = tuple(column for column in METADATA_COLUMNS if column not in _LABEL_COLUMNS)
    _require_columns(frame, required)
    present = [column for column in METADATA_COLUMNS if column in frame.columns]
    return cast(pd.DataFrame, frame.loc[:, present].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Serve features
# ---------------------------------------------------------------------------


def build_serve_features(
    moves: Iterable[ServeMove],
    *,
    player_elo: float,
    opponent_elo: float,
    time_control: str,
    elo_centering: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Materialize the same 12 columns for analyzed moves of one player.

    Forced moves and moves before ``MIN_HALF_MOVE`` are skipped because they are
    not model rows in training either.  If no move survives, an empty frame with
    the contract columns is returned so callers can treat "no analysable moves"
    as a first-class outcome.

    ``elo_centering`` is required in practice: it must be the mapping persisted
    from training.  Refitting here would center a live game on itself and break
    train/serve parity, so ``None`` fails closed.  Every move uses the player's
    derived rating bin (the same bins ChessFraud uses: 200 Elo wide, ``lt_1200``
    below 1200).
    """

    if elo_centering is None:
        raise FeatureContractError(
            "build_serve_features requires elo_centering fitted on the training split"
        )
    _validate_elo_centering(elo_centering)
    base_seconds = parse_time_control_base(time_control)
    player_rating = _require_finite_number("player_elo", player_elo)
    opponent_rating = _require_finite_number("opponent_elo", opponent_elo)

    kept = [move for move in moves if not move.is_forced and move.ply >= MIN_HALF_MOVE]
    if not kept:
        return _empty_feature_frame()

    rating_bin = _derived_rating_bin(player_rating)
    intermediate = pd.DataFrame(
        {
            "centipawn_loss": [
                max(0.0, float(move.eval_before_cp) - float(move.eval_played_cp))
                for move in kept
            ],
            "normalized_centipawn_loss": [
                normalized_centipawn_loss_from_evals(
                    float(move.eval_before_cp), float(move.eval_played_cp)
                )
                for move in kept
            ],
            "match_sf1": [float(move.is_top1_sf1) for move in kept],
            "match_sf9": [float(move.is_top1_sf9) for move in kept],
            "match_sf15": [float(move.is_top1_sf15) for move in kept],
            "eval_before": [float(move.eval_before_cp) for move in kept],
            "eval_after": [float(move.eval_played_cp) for move in kept],
            "half_move": [float(move.ply) for move in kept],
            "move_thinking_time": [
                0.0 if move.time_seconds is None else float(move.time_seconds)
                for move in kept
            ],
            "clock_remaining_time": [
                0.0 if move.clock_seconds is None else float(move.clock_seconds)
                for move in kept
            ],
            "base_seconds": base_seconds,
            "player_elo": player_rating,
            "opponent_elo": opponent_rating,
            "rating_bin": rating_bin,
        }
    )
    return _feature_matrix(intermediate, elo_centering)


# ---------------------------------------------------------------------------
# Internal: shared feature derivation
# ---------------------------------------------------------------------------


def _feature_matrix(
    intermediate: pd.DataFrame, elo_centering: Mapping[str, float]
) -> pd.DataFrame:
    """Derive the contract matrix from already-normalized intermediate columns."""

    _require_columns(intermediate, _INTERMEDIATE_COLUMNS)
    base_seconds = _to_float_column(intermediate, "base_seconds")
    if bool((base_seconds <= 0).any()):
        raise FeatureContractError("base_seconds must be positive for every row")

    thinking = _to_float_column(intermediate, "move_thinking_time", allow_missing=True)
    clock = _to_float_column(intermediate, "clock_remaining_time", allow_missing=True)
    eval_before = _to_float_column(intermediate, "eval_before")
    eval_after = _to_float_column(intermediate, "eval_after")

    features = pd.DataFrame(index=intermediate.index)
    features["centipawn_loss"] = _to_float_column(intermediate, "centipawn_loss").clip(
        lower=0.0, upper=MAX_CENTIPAWNS
    )
    features["normalized_centipawn_loss"] = _to_float_column(
        intermediate, "normalized_centipawn_loss"
    ).clip(lower=0.0, upper=1.0)
    features["match_sf1"] = _to_float_column(intermediate, "match_sf1")
    features["match_sf9"] = _to_float_column(intermediate, "match_sf9")
    features["match_sf15"] = _to_float_column(intermediate, "match_sf15")
    features["eval_swing"] = (eval_after - eval_before).clip(
        lower=-MAX_CENTIPAWNS, upper=MAX_CENTIPAWNS
    )
    features["abs_eval"] = eval_before.abs().clip(lower=0.0, upper=MAX_CENTIPAWNS)
    features["phase"] = (
        (_to_float_column(intermediate, "half_move") > float(ENDGAME_LAST_HALF_MOVE)).astype(int)
        + int(Phase.MIDDLEGAME)
    )
    features["move_thinking_time"] = thinking.clip(lower=0.0, upper=base_seconds)
    features["time_ratio"] = (clock / base_seconds).clip(lower=0.0, upper=MAX_TIME_RATIO)

    rating_bins = intermediate["rating_bin"].astype(str)
    features["player_elo_centered"] = _center_elo(
        _to_float_column(intermediate, "player_elo"), rating_bins, elo_centering, "player_elo"
    )
    features["opponent_elo_centered"] = _center_elo(
        _to_float_column(intermediate, "opponent_elo"), rating_bins, elo_centering, "opponent_elo"
    )

    ordered = cast(pd.DataFrame, features.loc[:, list(FEATURE_NAMES)].reset_index(drop=True))
    _validate_feature_frame(ordered)
    return ordered


def _validate_feature_frame(features: pd.DataFrame) -> None:
    """Validate columns/values with the contract, in bounded memory chunks."""

    validate_feature_columns(list(features.columns))
    if features.empty:
        validate_feature_values([])
        return
    for start in range(0, len(features), _VALIDATION_CHUNK_SIZE):
        chunk = features.iloc[start : start + _VALIDATION_CHUNK_SIZE]
        validate_feature_values(chunk.to_dict("records"))


def _empty_feature_frame() -> pd.DataFrame:
    empty = pd.DataFrame({name: pd.Series(dtype="float64") for name in FEATURE_NAMES})
    empty["phase"] = pd.Series(dtype="int64")
    return cast(pd.DataFrame, empty)


# ---------------------------------------------------------------------------
# Internal: input validation helpers
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureContractError(f"missing required source columns: {missing}")


def _reject_missing_text(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    counts = {column: int(frame[column].isna().sum()) for column in columns}
    missing = {column: count for column, count in counts.items() if count}
    if missing:
        raise FeatureContractError(
            f"used rows contain missing values in required inputs: {missing}"
        )


def _to_float_column(
    frame: pd.DataFrame, column: str, *, allow_missing: bool = False
) -> pd.Series:
    """Coerce a column to float, rejecting text garbage and (optionally) NaN."""

    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & values.isna()
    if bool(invalid.any()):
        raise FeatureContractError(f"column {column!r} contains non-numeric values")
    if bool(values.isna().any()):
        if not allow_missing:
            raise FeatureContractError(
                f"used rows contain missing values in required input {column!r}; "
                "drop those rows or pass allow_missing_timing=True"
            )
        values = values.fillna(0.0)
    return cast(pd.Series, values.astype(float))


def _require_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureContractError(f"{name} must be numeric, got {value!r}")
    if not isfinite(float(value)):
        raise FeatureContractError(f"{name} must be finite, got {value!r}")
    return float(value)


def _select_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Gate rows on ``is_used`` and the opening cutoff, fail closed on bad input."""

    _require_columns(df, ELIGIBILITY_COLUMNS)
    is_used = df["is_used"]
    if is_used.dtype == bool:
        used = is_used
    elif pd.api.types.is_integer_dtype(is_used) and set(pd.unique(is_used)) <= {0, 1}:
        used = is_used.astype(bool)
    else:
        raise FeatureContractError(
            "is_used must be a boolean (or 0/1 integer) column without missing values"
        )
    half_move = pd.to_numeric(df["half_move"], errors="coerce")
    if bool(half_move.isna().any()):
        raise FeatureContractError("half_move contains missing or non-numeric values")
    mask = used.to_numpy() & (half_move >= float(MIN_HALF_MOVE)).to_numpy()
    frame = cast(pd.DataFrame, df.loc[mask].copy())
    if frame.empty:
        raise FeatureContractError(
            f"no eligible rows: need is_used == True and half_move >= {MIN_HALF_MOVE}"
        )
    return frame


def _eligible_or_all(df: pd.DataFrame) -> pd.DataFrame:
    if all(column in df.columns for column in ELIGIBILITY_COLUMNS):
        return _select_eligible(df)
    return cast(pd.DataFrame, df.copy())


def _time_control_bases(time_control: pd.Series) -> pd.Series:
    unique_values = pd.unique(time_control)
    mapping = {value: parse_time_control_base(value) for value in unique_values}
    return cast(pd.Series, time_control.map(mapping).astype(float))


# ---------------------------------------------------------------------------
# Internal: Elo centering helpers
# ---------------------------------------------------------------------------


def _centering_key(kind: str, rating_bin: str) -> str:
    return f"{kind}{ELO_CENTERING_SEPARATOR}{rating_bin}"


def _derived_rating_bin(player_elo: float) -> str:
    if player_elo < 1200:
        return "lt_1200"
    low = int(player_elo // RATING_BIN_WIDTH) * RATING_BIN_WIDTH
    return f"{low}-{low + RATING_BIN_WIDTH}"


def _rating_bins(frame: pd.DataFrame, player_elo: pd.Series) -> pd.Series:
    derived = player_elo.map(_derived_rating_bin)
    if "rating_bin" not in frame.columns:
        return cast(pd.Series, derived.astype(str))
    raw = frame["rating_bin"].astype(object)
    text = raw.map(lambda value: "" if pd.isna(value) else str(value).strip())
    filled = text.where(text.str.len() > 0, derived)
    return cast(pd.Series, filled.astype(str))


def _fit_elo_centering_on_rows(
    frame: pd.DataFrame, player_elo: pd.Series, opponent_elo: pd.Series
) -> dict[str, float]:
    bins = _rating_bins(frame, player_elo)
    grouped = (
        pd.DataFrame({"__rating_bin": bins, "player_elo": player_elo, "opponent_elo": opponent_elo})
        .groupby("__rating_bin", sort=True)[["player_elo", "opponent_elo"]]
        .median()
    )
    centering: dict[str, float] = {}
    for bin_value, row in grouped.iterrows():
        centering[_centering_key("player_elo", str(bin_value))] = float(row["player_elo"])
        centering[_centering_key("opponent_elo", str(bin_value))] = float(row["opponent_elo"])
    centering[_centering_key("player_elo", GLOBAL_ELO_BIN)] = float(player_elo.median())
    centering[_centering_key("opponent_elo", GLOBAL_ELO_BIN)] = float(opponent_elo.median())
    return dict(sorted(centering.items()))


def _validate_elo_centering(centering: Mapping[str, float]) -> None:
    if not centering:
        raise FeatureContractError("elo_centering mapping must not be empty")
    for key, value in centering.items():
        if not isinstance(key, str):
            raise FeatureContractError(f"elo_centering keys must be strings, got {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FeatureContractError(f"elo_centering[{key!r}] must be numeric, got {value!r}")
        if not isfinite(float(value)):
            raise FeatureContractError(f"elo_centering[{key!r}] must be finite, got {value!r}")


def _center_elo(
    values: pd.Series,
    rating_bins: pd.Series,
    centering: Mapping[str, float],
    kind: str,
) -> pd.Series:
    keys = rating_bins.map(lambda rating_bin: _centering_key(kind, rating_bin))
    offsets = keys.map(centering)
    fallback = centering.get(_centering_key(kind, GLOBAL_ELO_BIN))
    if fallback is not None:
        offsets = offsets.fillna(float(fallback))
    if bool(offsets.isna().any()):
        missing = sorted({str(key) for key, value in offsets.items() if pd.isna(value)})
        raise FeatureContractError(
            f"elo_centering has no entry for {kind} rating bins {missing} "
            "and no __global__ fallback"
        )
    return cast(pd.Series, (values.astype(float) - offsets.astype(float)).astype(float))


__all__: Sequence[str] = (
    "METADATA_COLUMNS",
    "ServeMove",
    "build_serve_features",
    "build_training_features",
    "build_training_metadata",
    "fit_elo_centering",
    "normalized_centipawn_loss_from_evals",
    "parse_time_control_base",
)
