"""Bounded Stockfish adapter for per-move user analysis."""

import asyncio
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from .models import AnalysisConfig, MoveAnalysis
from .protocol import EngineAnalysisError


class StockfishEngine:
    """Run Stockfish with a fixed depth/time budget and no Maia dependency."""

    def __init__(
        self,
        executable: str | Path,
        *,
        depth: int = 15,
        time_ms: int = 100,
        engine_version: str = "unknown",
    ) -> None:
        self._config = AnalysisConfig(
            engine_name="stockfish",
            engine_version=engine_version,
            depth=depth,
            time_ms=time_ms,
        )
        self._executable = str(executable)

    @property
    def config(self) -> AnalysisConfig:
        return self._config

    async def analyze_game(self, pgn: str, username: str) -> Sequence[MoveAnalysis]:
        """Analyze only the requested player's moves in a PGN."""

        return await asyncio.to_thread(self._analyze_sync, pgn, username.casefold())

    def _analyze_sync(self, pgn: str, username: str) -> list[MoveAnalysis]:
        game = chess.pgn.read_game(StringIO(pgn))
        if game is None:
            raise EngineAnalysisError("PGN contains no game")
        board = game.board()
        moves: list[MoveAnalysis] = []
        try:
            with chess.engine.SimpleEngine.popen_uci(self._executable) as engine:
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    is_user_move = _move_belongs_to_user(game, board.turn, username)
                    if is_user_move:
                        fen_before = board.fen()
                        played_san = board.san(move)
                        info = engine.analyse(
                            board,
                            chess.engine.Limit(
                                depth=self._config.depth,
                                time=self._config.time_ms / 1000,
                            ),
                            multipv=3,
                        )
                        lines = info if isinstance(info, list) else [info]
                        best = lines[0]["pv"][0]
                        top_moves = {line["pv"][0] for line in lines if line.get("pv")}
                        eval_before = _score_cp(lines[0]["score"].pov(chess.WHITE))
                        best_board = board.copy(stack=False)
                        best_board.push(best)
                        best_info = engine.analyse(
                            best_board,
                            chess.engine.Limit(
                                depth=self._config.depth,
                                time=self._config.time_ms / 1000,
                            ),
                        )
                        eval_best = _score_cp(best_info["score"].pov(chess.WHITE))
                        board.push(move)
                        played_info = engine.analyse(
                            board,
                            chess.engine.Limit(
                                depth=self._config.depth,
                                time=self._config.time_ms / 1000,
                            ),
                        )
                        eval_played = _score_cp(played_info["score"].pov(chess.WHITE))
                        moves.append(
                            MoveAnalysis(
                                ply=ply,
                                fen_before=fen_before,
                                played_move_uci=move.uci(),
                                played_move_san=played_san,
                                eval_before_cp=eval_before,
                                eval_best_move_cp=eval_best,
                                eval_played_move_cp=eval_played,
                                is_top_1=move == best,
                                is_top_3=move in top_moves,
                                analysis_depth=self._config.depth,
                            )
                        )
                        continue
                    board.push(move)
        except (chess.engine.EngineError, OSError, ValueError) as exc:
            raise EngineAnalysisError(f"Stockfish analysis failed: {exc}") from exc
        if not moves:
            raise EngineAnalysisError("PGN contains no moves by requested user")
        return moves


def _move_belongs_to_user(game: chess.pgn.Game, turn: chess.Color, username: str) -> bool:
    header = game.headers.get("White" if turn == chess.WHITE else "Black", "")
    return header.casefold() == username


def _score_cp(score: chess.engine.PovScore) -> int:
    score_value = score.score(mate_score=100000)
    return int(score_value if score_value is not None else 0)
