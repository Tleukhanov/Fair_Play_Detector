"""Engine adapter protocol; concrete Stockfish integration is a later task."""

from collections.abc import Sequence
from typing import Protocol

from .models import AnalysisConfig, MoveAnalysis


class EngineAnalysisError(RuntimeError):
    """Raised when an engine cannot produce a valid bounded result."""


class ChessEngine(Protocol):
    """Minimal interface required by the analysis pipeline."""

    @property
    def config(self) -> AnalysisConfig: ...

    async def analyze_game(
        self,
        pgn: str,
        username: str,
    ) -> Sequence[MoveAnalysis]: ...
