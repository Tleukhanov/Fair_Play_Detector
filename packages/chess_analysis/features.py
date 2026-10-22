"""Deterministic aggregates derived from validated per-move analysis."""

from collections.abc import Sequence
from statistics import fmean, median, pvariance

from packages.feature_engineering.schema import GameAggregateFeatures

from .models import EVAL_CLIP_CP, MoveAnalysis


def centipawn_loss(best_move_cp: int, played_move_cp: int) -> float:
    """Return non-negative, clipped centipawn loss from mover-perspective evals."""

    return float(max(0, min(EVAL_CLIP_CP, best_move_cp - played_move_cp)))


def move_centipawn_loss(move: MoveAnalysis) -> float:
    """Prefer the adapter's explicit CPL and fall back to mover-POV evals."""

    if move.centipawn_loss is not None:
        return float(max(0.0, min(float(EVAL_CLIP_CP), move.centipawn_loss)))
    return centipawn_loss(move.eval_before_cp, move.eval_played_move_cp)


def aggregate_game_features(moves: Sequence[MoveAnalysis]) -> GameAggregateFeatures:
    """Aggregate usable moves while excluding forced/noisy positions."""

    usable = [move for move in moves if not move.is_forced and not move.is_noisy_position]
    if not usable:
        raise ValueError("at least one usable move is required")

    losses = [move_centipawn_loss(move) for move in usable]
    timed = [move.time_seconds for move in usable if move.time_seconds is not None]
    top_1_rate = sum(_is_top_1(usable_move) for usable_move in usable) / len(usable)
    top_3_rate = sum(move.is_top_3 for move in usable) / len(usable)
    identical_time_share = _identical_time_share(timed)
    time_variance = pvariance(timed) if len(timed) > 1 else None

    return GameAggregateFeatures(
        move_count=len(usable),
        mean_centipawn_loss=fmean(losses),
        median_centipawn_loss=median(losses),
        p90_centipawn_loss=_percentile(losses, 0.9),
        top_1_match_rate=top_1_rate,
        top_3_match_rate=top_3_rate,
        difficult_position_match_rate=top_1_rate,
        tactical_error_count=sum(loss >= 100 for loss in losses),
        centipawn_loss_stability=pvariance(losses) if len(losses) > 1 else 0.0,
        identical_time_share=identical_time_share,
        median_time_seconds=median(timed) if timed else None,
        mean_time_seconds=fmean(timed) if timed else None,
        time_variance=time_variance,
        time_difficulty_correlation=None,
        maia_divergence=_maia_divergence(usable),
    )


def _is_top_1(move: MoveAnalysis) -> bool:
    """Treat ``is_top_1`` as a legacy alias of the main-depth ``is_top1_sf15``."""

    return bool(move.is_top1_sf15 or move.is_top_1)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def _identical_time_share(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    most_common = max(values.count(value) for value in set(values))
    return most_common / len(values)


def _maia_divergence(moves: Sequence[MoveAnalysis]) -> float | None:
    probabilities = [
        move.maia_move_probability
        for move in moves
        if move.maia_move_probability is not None
    ]
    if not probabilities:
        return None
    return 1.0 - fmean(probabilities)
