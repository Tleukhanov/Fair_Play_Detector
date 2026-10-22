"""Loading and filtering helpers for the pinned ChessFraud parquet tables.

The downloader (``scripts/download_chessfraud.py``) stores the tables in
``data_raw/`` by default and records their checksums in
``data_raw/manifest.json``.  This module only reads those local files; it never
downloads anything, which keeps training and evaluation reproducible from a
verified snapshot.

Set ``FAIR_PLAY_DATA_DIR`` to point at a different directory (for example a
shared cache).  Relative paths are resolved against the current working
directory, while the default is ``<repo>/data_raw``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as parquet  # type: ignore[import-untyped]

from packages.ml.feature_contract import (
    GROUP_COLUMN,
    MIN_HALF_MOVE,
    validate_player_disjoint,
)

DATA_DIR_ENV_VAR = "FAIR_PLAY_DATA_DIR"
DEFAULT_DATA_DIR = "data_raw"

CHESSFRAUD_FILENAME = "chessfraud_full.parquet"
SYNTH_TRAIN_FILENAME = "synth_train.parquet"
SYNTH_TEST_FILENAME = "synth_test.parquet"

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Columns shared by every table that downstream feature construction relies on.
_COMMON_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "player_id",
        "opponent_id",
        "player_color",
        "player_elo",
        "opponent_elo",
        "time_control",
        "game_result",
        "half_move",
        "move_player",
        "is_used",
        "eval_before",
        "eval_after",
        "centipawn_loss",
        "normalized_centipawn_loss",
        "move_thinking_time",
        "clock_remaining_time",
        "move_stockfish_1",
        "move_stockfish_9",
        "move_stockfish_15",
    }
)

CHESSFRAUD_REQUIRED_COLUMNS: frozenset[str] = _COMMON_COLUMNS | {
    "tournament_id",
    "player_hint_shown",
    "assistance_search_depth",
    "assistance_line_rank",
    "is_cheating_move",
    "is_cheating_player_game",
}

SYNTH_REQUIRED_COLUMNS: frozenset[str] = _COMMON_COLUMNS | {
    "rating_bin",
    "move_stockfish_5",
    "move_stockfish_11",
}

__all__ = [
    "CHESSFRAUD_FILENAME",
    "CHESSFRAUD_REQUIRED_COLUMNS",
    "DATA_DIR_ENV_VAR",
    "DatasetSchemaError",
    "SYNTH_REQUIRED_COLUMNS",
    "SYNTH_TEST_FILENAME",
    "SYNTH_TRAIN_FILENAME",
    "assert_player_disjoint",
    "filter_opening",
    "filter_used",
    "get_data_dir",
    "load_chessfraud",
    "load_synth_test",
    "load_synth_train",
    "resolve_data_path",
    "summarize",
]


class DatasetSchemaError(ValueError):
    """Raised when a parquet table does not match the expected ChessFraud schema."""


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the dataset directory, honoring ``FAIR_PLAY_DATA_DIR``."""

    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return _REPO_ROOT / DEFAULT_DATA_DIR


def resolve_data_path(
    filename: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Return the absolute-or-relative path of one table inside the data dir."""

    return get_data_dir(data_dir) / filename


def _read_table(
    filename: str,
    required_columns: frozenset[str],
    *,
    data_dir: str | Path | None,
) -> pd.DataFrame:
    path = resolve_data_path(filename, data_dir=data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"ChessFraud table not found: {path}. "
            "Run `python scripts/download_chessfraud.py` first or set "
            f"{DATA_DIR_ENV_VAR}."
        )
    frame = parquet.read_table(path).to_pandas()
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise DatasetSchemaError(f"{path.name} is missing required columns: {', '.join(missing)}")
    return frame


def load_chessfraud(*, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the controlled tournament table (``chess_fraud/full``)."""

    return _read_table(
        CHESSFRAUD_FILENAME,
        CHESSFRAUD_REQUIRED_COLUMNS,
        data_dir=data_dir,
    )


def load_synth_train(*, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the ChessFraud-Synth training split."""

    return _read_table(SYNTH_TRAIN_FILENAME, SYNTH_REQUIRED_COLUMNS, data_dir=data_dir)


def load_synth_test(*, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the ChessFraud-Synth player-disjoint test split."""

    return _read_table(SYNTH_TEST_FILENAME, SYNTH_REQUIRED_COLUMNS, data_dir=data_dir)


def filter_used(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only paper-aligned decision points (``is_used == True``)."""

    return frame.loc[frame["is_used"].eq(True)].copy()


def filter_opening(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop opening moves below the shared ``MIN_HALF_MOVE`` cutoff."""

    return frame.loc[frame["half_move"] >= MIN_HALF_MOVE].copy()


def assert_player_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any player appears in both splits."""

    validate_player_disjoint(
        train[GROUP_COLUMN].unique(),
        test[GROUP_COLUMN].unique(),
        split_name="train/test",
    )


def _fraction(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    return float(frame[column].eq(True).mean())


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Return row/player/game counts, Elo range, and available label rates.

    ChessFraud-Synth has no stored labels (fair/assisted pairs are reconstructed
    per source), so both cheat rates are ``None`` for those tables.
    """

    rows = int(len(frame))
    elo_range: dict[str, int | None]
    if rows:
        players = int(frame[GROUP_COLUMN].nunique())
        games = int(frame["game_id"].nunique())
        elo = frame["player_elo"].dropna()
        if elo.empty:
            elo_range = {"min": None, "max": None}
        else:
            elo_range = {"min": int(elo.min()), "max": int(elo.max())}
    else:
        players = 0
        games = 0
        elo_range = {"min": None, "max": None}

    cheat_game_rate: float | None = None
    if rows and "is_cheating_player_game" in frame.columns:
        games_frame = frame.drop_duplicates(subset=["game_id"])
        cheat_game_rate = float(games_frame["is_cheating_player_game"].eq(True).mean())

    return {
        "rows": rows,
        "players": players,
        "games": games,
        "elo_range": elo_range,
        "cheat_move_rate": _fraction(frame, "is_cheating_move"),
        "cheat_game_rate": cheat_game_rate,
    }
