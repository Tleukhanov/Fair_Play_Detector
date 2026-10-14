"""Chess engine analysis contracts and feature aggregation."""

from .features import aggregate_game_features, centipawn_loss
from .models import AnalysisConfig, MoveAnalysis
from .protocol import ChessEngine, EngineAnalysisError

__all__ = [
    "AnalysisConfig",
    "ChessEngine",
    "EngineAnalysisError",
    "MoveAnalysis",
    "aggregate_game_features",
    "centipawn_loss",
]
