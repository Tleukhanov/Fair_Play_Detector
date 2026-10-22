"""Typed analysis records shared by engine adapters and feature engineering."""

from dataclasses import dataclass

#: Evaluation bound in centipawns. Mate scores are mapped onto this value so a
#: single constant caps both plain evaluations and forced mates.
EVAL_CLIP_CP = 1000


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
    """One user move and the engine alternatives used to score it.

    All ``*_cp`` fields are evaluated from the point of view of the player who
    made the move (``mover``), clipped to ``[-1000, 1000]``; a forced mate maps
    onto ``+/-1000``. ``centipawn_loss`` is the mover-perspective drop caused by
    the played move and is clipped to ``[0, 1000]``.

    ``is_top1_sf1``/``is_top1_sf9``/``is_top1_sf15`` record an exact top-1 match
    (same UCI move) between the played move and the engine's first choice at
    depth 1, depth 9, and the configured main depth (15 by default).
    ``is_top_1`` is a backwards-compatible alias of ``is_top1_sf15``.

    ``is_forced`` marks positions with exactly one legal move. A move is
    ``is_noisy_position`` when the engine already sees a forced mate for either
    side before the move; such moves carry no useful signal and are filtered out
    of aggregates. ``time_seconds`` is the clock delta for the move and is
    ``None`` when the PGN has no ``%clk`` annotations.
    """

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
    centipawn_loss: float | None = None
    is_top1_sf1: bool = False
    is_top1_sf9: bool = False
    is_top1_sf15: bool = False

    def __post_init__(self) -> None:
        for name in ("eval_before_cp", "eval_best_move_cp", "eval_played_move_cp"):
            value = getattr(self, name)
            if not -EVAL_CLIP_CP <= value <= EVAL_CLIP_CP:
                raise ValueError(f"{name} must be clipped to [-{EVAL_CLIP_CP}, {EVAL_CLIP_CP}]")
        if self.centipawn_loss is not None and not (
            0 <= self.centipawn_loss <= EVAL_CLIP_CP
        ):
            raise ValueError(
                f"centipawn_loss must be within [0, {EVAL_CLIP_CP}]"
            )
        if self.time_seconds is not None and self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
