"""Chess engine analysis contracts and feature aggregation."""

from .features import aggregate_game_features, centipawn_loss
from .models import AnalysisConfig, MoveAnalysis
from .protocol import ChessEngine, EngineAnalysisError
from .stockfish import StockfishEngine

__all__ = [
    "AnalysisConfig",
    "ChessEngine",
    "EngineAnalysisError",
    "MoveAnalysis",
    "StockfishEngine",
    "aggregate_game_features",
    "centipawn_loss",
]
