"""Typed analysis records shared by engine adapters and feature engineering."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Bounded engine budget stored with every analysis result."""

    engine_name: str
    engine_version: str
    depth: int
    time_ms: int

    def __post_init__(self) -> None:
        if self.depth < 1 or self.time_ms < 1:
            raise ValueError("analysis depth and time_ms must be positive")


@dataclass(frozen=True, slots=True)
class MoveAnalysis:
    """One user move and the engine alternatives used to score it."""

    ply: int
    fen_before: str
    played_move_uci: str
    played_move_san: str
    eval_before_cp: int
    eval_best_move_cp: int
    eval_played_move_cp: int
    is_top_1: bool
    is_top_3: bool
    time_seconds: float | None = None
    analysis_depth: int | None = None
    maia_eval: float | None = None
    maia_move_probability: float | None = None
    is_forced: bool = False
    is_noisy_position: bool = False
