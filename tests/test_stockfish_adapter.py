"""Integration tests for the Stockfish adapter's per-move semantics."""

import asyncio
import shutil

import pytest

from packages.chess_analysis import MoveAnalysis, StockfishEngine, aggregate_game_features

STOCKFISH = shutil.which("stockfish") or "/home/yeraly/.local/bin/stockfish"

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="stockfish binary is not installed",
)

DEPTH = 6
TIME_MS = 200


def _analyze(pgn: str, username: str) -> list[MoveAnalysis]:
    engine = StockfishEngine(STOCKFISH, depth=DEPTH, time_ms=TIME_MS)
    return list(asyncio.run(engine.analyze_game(pgn, username)))


WHITE_BLUNDER = """[Event "white blunder"]
[Site "https://lichess.org/whiteblunder"]
[GameId "whiteblunder"]
[Variant "Standard"]
[Result "0-1"]
[White "Blunderer"]
[Black "Opponent"]
[TimeControl "300+0"]

1. f3 e5 2. g4 Qh4# 0-1
"""

BLACK_BLUNDER = """[Event "black blunder"]
[Site "https://lichess.org/blackblunder"]
[GameId "blackblunder"]
[Variant "Standard"]
[Result "1-0"]
[White "Opponent"]
[Black "Blunderer"]
[TimeControl "300+0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

CLOCKED_GAME = """[Event "clocked"]
[Site "https://lichess.org/clocked"]
[GameId "clocked"]
[Variant "Standard"]
[Result "1-0"]
[White "ClockUser"]
[Black "Other"]
[TimeControl "300+3"]

1. e4 {[%clk 0:04:58]} e5 {[%clk 0:04:59]} 2. Nf3 {[%clk 0:04:55]} Nc6 {[%clk 0:04:57]} 1-0
"""

MATE_IN_ONE = """[Event "mate in one"]
[Site "https://lichess.org/mate"]
[GameId "mate"]
[Variant "Standard"]
[Result "1-0"]
[White "Mater"]
[Black "Other"]
[TimeControl "300+0"]
[SetUp "1"]
[FEN "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"]

1. Qg7# 1-0
"""

FORCED_MOVE = """[Event "forced"]
[Site "https://lichess.org/forced"]
[GameId "forced"]
[Variant "Standard"]
[Result "*"]
[White "Forced"]
[Black "Other"]
[TimeControl "300+0"]
[SetUp "1"]
[FEN "7r/8/8/8/8/5k2/8/7K w - - 0 1"]

1. Kg1 *
"""

HANGING_QUEEN = """[Event "hanging queen"]
[Site "https://lichess.org/trivial"]
[GameId "trivial"]
[Variant "Standard"]
[Result "*"]
[White "Trivial"]
[Black "Other"]
[TimeControl "300+0"]
[SetUp "1"]
[FEN "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"]

1. exd5 *
"""


def test_white_blunder_has_positive_centipawn_loss() -> None:
    moves = _analyze(WHITE_BLUNDER, "Blunderer")

    assert [move.played_move_san for move in moves] == ["f3", "g4"]
    blunder = moves[-1]
    # The old adapter returned 0 for every Black move and a White-POV value for
    # White; g4 walks into mate, so its loss must be large and non-zero.
    assert blunder.centipawn_loss is not None
    assert blunder.centipawn_loss > 500
    assert blunder.eval_before_cp > blunder.eval_played_move_cp
    assert blunder.eval_played_move_cp == -1000


def test_black_blunder_has_positive_centipawn_loss() -> None:
    moves = _analyze(BLACK_BLUNDER, "Blunderer")

    by_san = {move.played_move_san: move for move in moves}
    blunder = by_san["Nf6"]
    # Regression for the White-POV bug: Black's blunder used to score CPL 0.
    assert blunder.centipawn_loss is not None
    assert blunder.centipawn_loss > 500
    assert blunder.eval_played_move_cp == -1000


def test_mate_scores_are_clipped_to_one_thousand() -> None:
    move = _analyze(MATE_IN_ONE, "Mater")[0]

    assert move.played_move_san == "Qg7#"
    assert move.eval_before_cp == 1000
    assert move.eval_played_move_cp == 1000
    assert move.centipawn_loss == 0.0
    assert -1000 <= move.eval_before_cp <= 1000
    assert -1000 <= move.eval_best_move_cp <= 1000


def test_forced_move_is_flagged() -> None:
    move = _analyze(FORCED_MOVE, "Forced")[0]

    assert move.played_move_san == "Kg1"
    assert move.is_forced is True


def test_clocks_produce_positive_thinking_time() -> None:
    moves = _analyze(CLOCKED_GAME, "ClockUser")

    times = [move.time_seconds for move in moves]
    assert times == [2.0, 3.0]
    assert all(value is not None and value > 0 for value in times)


def test_missing_clocks_leave_thinking_time_none() -> None:
    moves = _analyze(WHITE_BLUNDER, "Blunderer")

    assert all(move.time_seconds is None for move in moves)


def test_depth_agreement_fields_and_top1() -> None:
    move = _analyze(HANGING_QUEEN, "Trivial")[0]

    assert move.played_move_san == "exd5"
    assert move.is_top1_sf1 is True
    assert move.is_top1_sf9 is True
    assert move.is_top1_sf15 is True
    assert move.is_top_1 is True
    assert move.is_top_3 is True


def test_aggregate_uses_explicit_cpl_and_skips_forced_moves() -> None:
    blunder_moves = _analyze(BLACK_BLUNDER, "Blunderer")
    forced_move = _analyze(FORCED_MOVE, "Forced")[0]

    aggregate = aggregate_game_features([*blunder_moves, forced_move])

    # The forced root position is excluded, so only Black's three moves remain.
    assert aggregate.move_count == 3
    assert aggregate.mean_centipawn_loss > 0
    assert forced_move.is_forced is True


def test_engine_version_is_reported_from_uci() -> None:
    engine = StockfishEngine(STOCKFISH, depth=DEPTH, time_ms=TIME_MS)

    assert engine.config.engine_version.lower().startswith("stockfish")
