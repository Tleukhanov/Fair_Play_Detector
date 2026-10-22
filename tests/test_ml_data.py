"""Tests for the local ChessFraud loaders and downloader manifest helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from packages.ml.data import (
    CHESSFRAUD_FILENAME,
    CHESSFRAUD_REQUIRED_COLUMNS,
    SYNTH_REQUIRED_COLUMNS,
    SYNTH_TEST_FILENAME,
    SYNTH_TRAIN_FILENAME,
    DatasetSchemaError,
    assert_player_disjoint,
    filter_opening,
    filter_used,
    load_chessfraud,
    load_synth_test,
    load_synth_train,
    summarize,
)
from packages.ml.feature_contract import (
    GROUP_COLUMN,
    MIN_HALF_MOVE,
    FeatureContractError,
)


def _load_downloader() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_chessfraud.py"
    spec = importlib.util.spec_from_file_location("download_chessfraud", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "game-1",
        "player_id": "player-1",
        "opponent_id": "player-2",
        "player_color": "white",
        "player_elo": 1800,
        "opponent_elo": 1750,
        "time_control": "300+0",
        "game_result": "1-0",
        "half_move": 22,
        "move_player": "e2e4",
        "position_before": "fen-before",
        "position_after": "fen-after",
        "is_used": True,
        "eval_before": 20,
        "eval_after": 15,
        "centipawn_loss": 5,
        "normalized_centipawn_loss": 0.01,
        "move_thinking_time": 3.5,
        "clock_remaining_time": 250.0,
        "move_stockfish_1": "e2e4",
        "move_stockfish_5": "e2e4",
        "move_stockfish_9": "d2d4",
        "move_stockfish_11": "d2d4",
        "move_stockfish_15": "g1f3",
        "tournament_id": 4,
        "player_hint_shown": False,
        "assistance_search_depth": 14,
        "assistance_line_rank": 1,
        "is_cheating_move": True,
        "is_cheating_player_game": True,
        "rating_bin": "1800-2000",
    }
    row.update(overrides)
    return row


def _chessfraud_row(**overrides: object) -> dict[str, object]:
    row = _row(**overrides)
    return {column: row[column] for column in CHESSFRAUD_REQUIRED_COLUMNS}


def _synth_row(**overrides: object) -> dict[str, object]:
    row = _row(**overrides)
    return {column: row[column] for column in SYNTH_REQUIRED_COLUMNS}


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    parquet.write_table(pa.Table.from_pylist(rows), path)


def test_sha256_and_manifest_round_trip(tmp_path: Path) -> None:
    downloader = _load_downloader()
    payload = b"chessfraud-fixture"
    data_file = tmp_path / "tiny.bin"
    data_file.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert downloader.sha256_file(data_file) == expected

    manifest = {
        "dataset": "artemlepin/chess-fraud",
        "revision": "0" * 40,
        "files": [
            {
                "name": "tiny.bin",
                "bytes": len(payload),
                "sha256": expected,
                "url": "https://example.invalid/tiny.bin",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    downloader.write_manifest(manifest_path, manifest)

    assert downloader.read_manifest(manifest_path) == manifest
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")


def test_required_column_validation_raises(tmp_path: Path) -> None:
    rows = [_chessfraud_row()]
    del rows[0]["is_cheating_move"]
    _write_table(tmp_path / CHESSFRAUD_FILENAME, rows)

    with pytest.raises(DatasetSchemaError, match="is_cheating_move"):
        load_chessfraud(data_dir=tmp_path)


def test_missing_table_reports_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_chessfraud.py"):
        load_chessfraud(data_dir=tmp_path)


def test_filter_used_and_opening() -> None:
    frame = pd.DataFrame(
        {
            "is_used": [True, False, True, True],
            "half_move": [MIN_HALF_MOVE - 1, 30, MIN_HALF_MOVE, 40],
        }
    )

    used = filter_used(frame)
    assert used["is_used"].tolist() == [True, True, True]

    non_opening = filter_opening(frame)
    assert non_opening["half_move"].tolist() == [30, MIN_HALF_MOVE, 40]

    both = filter_opening(filter_used(frame))
    assert both["half_move"].tolist() == [MIN_HALF_MOVE, 40]
    assert len(frame) == 4


def test_player_disjoint_guard_raises_on_overlap() -> None:
    train = pd.DataFrame({GROUP_COLUMN: ["alice", "bob"]})
    test = pd.DataFrame({GROUP_COLUMN: ["bob", "carol"]})

    with pytest.raises(FeatureContractError, match="not player-disjoint"):
        assert_player_disjoint(train, test)

    assert_player_disjoint(train, pd.DataFrame({GROUP_COLUMN: ["carol"]}))


def test_loaders_and_summary_on_tiny_parquet(tmp_path: Path) -> None:
    _write_table(tmp_path / CHESSFRAUD_FILENAME, [_chessfraud_row()])
    _write_table(
        tmp_path / SYNTH_TRAIN_FILENAME,
        [_synth_row(player_id="synth-train", game_id="synth-train-game")],
    )
    _write_table(
        tmp_path / SYNTH_TEST_FILENAME,
        [_synth_row(player_id="synth-test", game_id="synth-test-game")],
    )

    chessfraud = load_chessfraud(data_dir=tmp_path)
    train = load_synth_train(data_dir=tmp_path)
    test = load_synth_test(data_dir=tmp_path)

    assert len(chessfraud) == len(train) == len(test) == 1
    assert set(chessfraud.columns) == CHESSFRAUD_REQUIRED_COLUMNS
    assert set(train.columns) == SYNTH_REQUIRED_COLUMNS

    summary = summarize(chessfraud)
    assert summary["rows"] == 1
    assert summary["players"] == 1
    assert summary["games"] == 1
    assert summary["elo_range"] == {"min": 1800, "max": 1800}
    assert summary["cheat_move_rate"] == 1.0
    assert summary["cheat_game_rate"] == 1.0

    synth_summary = summarize(train)
    assert synth_summary["cheat_move_rate"] is None
    assert synth_summary["cheat_game_rate"] is None

    assert_player_disjoint(train, test)


def test_env_var_selects_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_table(tmp_path / CHESSFRAUD_FILENAME, [_chessfraud_row()])
    monkeypatch.setenv("FAIR_PLAY_DATA_DIR", str(tmp_path))

    assert len(load_chessfraud()) == 1
