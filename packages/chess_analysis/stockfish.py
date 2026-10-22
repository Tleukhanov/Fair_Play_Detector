"""Bounded Stockfish adapter for per-move user analysis.

Evaluation semantics: every centipawn value stored on :class:`MoveAnalysis` is
from the point of view of the player who made the move and is clipped to
``[-1000, 1000]`` with forced mates mapped onto the bounds.

Each analysed user move costs three engine calls on the position before the move
(multipv=3 at the main depth, plus depth-1 and depth-9 top-1 probes) and one call
on the position after the played move; the best line already provides the
"evaluation after the best move", so no third full-depth call is needed.
"""

import asyncio
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from .models import EVAL_CLIP_CP, AnalysisConfig, MoveAnalysis
from .protocol import EngineAnalysisError

#: Depth used for the cheap "did the player find the shallow engine move" probes.
SHALLOW_DEPTH = 9


class StockfishEngine:
    """Run Stockfish with a fixed depth/time budget and no Maia dependency."""

    def __init__(
        self,
        executable: str | Path,
        *,
        depth: int = 15,
        time_ms: int = 100,
        engine_version: str | None = None,
    ) -> None:
        self._executable = str(executable)
        self._config = AnalysisConfig(
            engine_name="stockfish",
            engine_version=engine_version or _query_engine_version(self._executable),
            depth=depth,
            time_ms=time_ms,
        )

    @property
    def config(self) -> AnalysisConfig:
        return self._config

    async def analyze_game(self, pgn: str, username: str) -> Sequence[MoveAnalysis]:
        """Analyze only the requested player's moves in a PGN."""

        return await asyncio.to_thread(self._analyze_sync, pgn, username.casefold())

    def _analyze_sync(self, pgn: str, username: str) -> list[MoveAnalysis]:
        """Analyze one game synchronously, raising on any engine failure."""

        game = chess.pgn.read_game(StringIO(pgn))
        if game is None:
            raise EngineAnalysisError("PGN contains no game")
        board = game.board()
        base_seconds = _parse_time_control(game.headers.get("TimeControl", ""))
        # Clocks alternate, so elapsed time is tracked per colour.
        previous_clocks: dict[chess.Color, float | None] = {
            chess.WHITE: base_seconds,
            chess.BLACK: base_seconds,
        }
        moves: list[MoveAnalysis] = []
        try:
            with chess.engine.SimpleEngine.popen_uci(self._executable) as engine:
                for ply, node in enumerate(game.mainline(), start=1):
                    move = node.move
                    remaining = node.clock()
                    thinking = _thinking_seconds(
                        previous_clocks[board.turn], remaining, base_seconds
                    )
                    if remaining is not None:
                        # Missing clocks are never invented; the next observed
                        # clock simply spans the gap.
                        previous_clocks[board.turn] = remaining
                    if not _move_belongs_to_user(game, board.turn, username):
                        board.push(move)
                        continue

                    mover = board.turn
                    fen_before = board.fen()
                    played_san = board.san(move)
                    is_forced = board.legal_moves.count() == 1

                    # (a) Position before the move: top-3 at the main depth, plus
                    # the depth-1 and depth-9 top-1 probes for parity features.
                    main_lines = _analyse_lines(
                        engine,
                        board,
                        multipv=3,
                        limit=self._main_limit(),
                    )
                    best_line = main_lines[0]
                    best_move = _pv_head(best_line)
                    top_moves = {
                        head for head in (_pv_head(line) for line in main_lines) if head
                    }
                    eval_before = _mover_cp(best_line["score"], mover)
                    is_noisy = bool(best_line["score"].pov(mover).is_mate())
                    top1_sf1 = move == _pv_head(
                        engine.analyse(board, chess.engine.Limit(depth=1))
                    )
                    top1_sf9 = move == _pv_head(
                        engine.analyse(board, chess.engine.Limit(depth=SHALLOW_DEPTH))
                    )

                    board.push(move)
                    # (b) Position after the played move, from the mover's POV.
                    eval_played = self._eval_after_played_move(engine, board, mover)
                    cpl = _clip_loss(eval_before - eval_played)

                    moves.append(
                        MoveAnalysis(
                            ply=ply,
                            fen_before=fen_before,
                            played_move_uci=move.uci(),
                            played_move_san=played_san,
                            eval_before_cp=eval_before,
                            # The best line's own evaluation already describes
                            # the position after the engine's best move.
                            eval_best_move_cp=eval_before,
                            eval_played_move_cp=eval_played,
                            is_top_1=move == best_move,
                            is_top_3=move in top_moves,
                            is_top1_sf1=top1_sf1,
                            is_top1_sf9=top1_sf9,
                            is_top1_sf15=move == best_move,
                            centipawn_loss=cpl,
                            time_seconds=thinking,
                            analysis_depth=self._config.depth,
                            is_forced=is_forced,
                            is_noisy_position=is_noisy,
                        )
                    )
        except (chess.engine.EngineError, OSError, ValueError) as exc:
            raise EngineAnalysisError(f"Stockfish analysis failed: {exc}") from exc
        if not moves:
            raise EngineAnalysisError("PGN contains no moves by requested user")
        return moves

    def _main_limit(self) -> chess.engine.Limit:
        return chess.engine.Limit(
            depth=self._config.depth,
            time=self._config.time_ms / 1000,
        )

    def _eval_after_played_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        mover: chess.Color,
    ) -> int:
        """Return the mover-POV evaluation of the position after the played move."""

        if board.is_checkmate():
            return EVAL_CLIP_CP
        if board.is_game_over(claim_draw=False):
            return 0
        info = engine.analyse(board, self._main_limit())
        return _mover_cp(info["score"], mover)


def _query_engine_version(executable: str) -> str:
    """Read the UCI ``id name`` of the given Stockfish binary."""

    try:
        with chess.engine.SimpleEngine.popen_uci(executable) as engine:
            name = engine.id.get("name")
    except (chess.engine.EngineError, OSError) as exc:
        raise EngineAnalysisError(f"cannot query Stockfish version: {exc}") from exc
    return str(name) if name else "unknown"


def _analyse_lines(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    multipv: int,
    limit: chess.engine.Limit,
) -> list[chess.engine.InfoDict]:
    info = engine.analyse(board, limit, multipv=multipv)
    return list(info) if isinstance(info, list) else [info]


def _pv_head(info: chess.engine.InfoDict) -> chess.Move | None:
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0]


def _mover_cp(score: chess.engine.PovScore, mover: chess.Color) -> int:
    pov = score.pov(mover)
    mate = pov.mate()
    if mate is not None:
        # Any forced mate collapses onto the clip bound instead of losing
        # strength with the mate distance.
        return EVAL_CLIP_CP if mate > 0 else -EVAL_CLIP_CP
    value = pov.score(mate_score=EVAL_CLIP_CP)
    return _clip(int(value if value is not None else 0))


def _clip(value: int) -> int:
    return max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, value))


def _clip_loss(value: int) -> float:
    return float(max(0, min(EVAL_CLIP_CP, value)))


def _move_belongs_to_user(game: chess.pgn.Game, turn: chess.Color, username: str) -> bool:
    header = game.headers.get("White" if turn == chess.WHITE else "Black", "")
    return header.casefold() == username


def _parse_time_control(value: str) -> float | None:
    """Return the base time in seconds from a Lichess ``base+increment`` header."""

    if not value or value == "-":
        return None
    base, _separator, _increment = value.partition("+")
    try:
        seconds = float(base)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _thinking_seconds(
    previous_remaining: float | None,
    remaining: float | None,
    base_seconds: float | None,
) -> float | None:
    """Return seconds spent on the move, clipped to ``[0, base]`` when known."""

    if remaining is None or previous_remaining is None:
        return None
    spent = max(0.0, previous_remaining - remaining)
    if base_seconds is not None:
        spent = min(spent, base_seconds)
    return round(spent, 3)
