"""Deterministic baseline scorer used before a trained model is available."""

from collections.abc import Iterable
from math import sqrt

from .contracts import (
    ContributionDirection,
    FeatureEvidence,
    ModelMetadata,
    RiskThresholds,
    ScoreResult,
    ScoringInput,
)


class InsufficientGamesError(ValueError):
    """Raised when a score would violate the minimum-sample policy."""


class BaselineScorer:
    """Conservative heuristic scorer with explicit, reviewable weights.

    Each evidence item also carries a signed ``contribution`` relative to the
    evidence baseline: ``weight * (signal - baseline_signal)``.  This keeps the
    explanation causal in the heuristic's own scoring rule while leaving the risk
    score itself unchanged.  Contributions are heuristic deltas on the score
    scale, not calibrated probabilities.
    """

    minimum_games = 10

    def __init__(self, metadata: ModelMetadata) -> None:
        self._metadata = metadata

    def score(self, scoring_input: ScoringInput) -> ScoreResult:
        games = scoring_input.games
        if len(games) < self.minimum_games:
            raise InsufficientGamesError(
                f"at least {self.minimum_games} usable games are required for scoring"
            )

        top_move_rate = _mean(game.top_1_match_rate for game in games)
        cpl_signal = _inverse_normalized_mean(
            (game.mean_centipawn_loss for game in games), maximum=100.0
        )
        time_signal = _mean(game.identical_time_share for game in games)
        maia_values = [game.maia_divergence for game in games if game.maia_divergence is not None]
        maia_signal = _mean(maia_values) if maia_values else 0.0

        risk_score = _clamp(
            0.35 * top_move_rate
            + 0.30 * cpl_signal
            + 0.20 * time_signal
            + 0.15 * maia_signal
        )
        evidence: list[FeatureEvidence] = [
            FeatureEvidence(
                code="HIGH_TOP_MOVE_RATE",
                label="High top-1 move match rate",
                value=top_move_rate,
                baseline=0.61,
                weight=0.35,
            ),
            FeatureEvidence(
                code="LOW_CENTIPAWN_LOSS",
                label="Low mean centipawn loss",
                value=_mean(game.mean_centipawn_loss for game in games),
                baseline=35.0,
                weight=0.30,
            ),
            FeatureEvidence(
                code="REGULAR_MOVE_TIMES",
                label="Similar time spent per move",
                value=time_signal,
                baseline=0.20,
                weight=0.20,
            ),
            FeatureEvidence(
                code="MAIA_DIVERGENCE",
                label="Divergence from Maia baseline",
                value=maia_signal,
                baseline=0.15,
                weight=0.15,
            ),
        ]
        evidence = _with_contributions(evidence)
        return ScoreResult(
            risk_level=_risk_level(risk_score, self._metadata.thresholds),
            risk_score=risk_score,
            confidence_interval=_confidence_interval(risk_score, len(games)),
            games_analyzed=len(games),
            evidence=evidence,
            model=self._metadata,
        )


def _with_contributions(evidence: list[FeatureEvidence]) -> list[FeatureEvidence]:
    """Attach weighted deltas from the evidence baseline.

    The centipawn-loss evidence stores raw CPL, while the score uses inverse CPL
    ``1 - cpl / 100``; the other evidence values are already the scored signal.
    """

    contributions: list[float] = []
    for item in evidence:
        if item.code == "LOW_CENTIPAWN_LOSS":
            signal = 1.0 - (item.value / 100.0)
            baseline_signal = 1.0 - (item.baseline / 100.0)
        else:
            signal = item.value
            baseline_signal = item.baseline
        contributions.append(item.weight * (signal - baseline_signal))
    total = sum(abs(value) for value in contributions)
    enriched: list[FeatureEvidence] = []
    for item, contribution in zip(evidence, contributions, strict=True):
        share = abs(contribution) / total if total > 0.0 else 0.0
        direction = (
            ContributionDirection.SUPPORTS_RISK
            if contribution > 0.0
            else ContributionDirection.REDUCES_RISK
            if contribution < 0.0
            else ContributionDirection.NEUTRAL
        )
        enriched.append(
            item.model_copy(
                update={
                    "contribution": contribution,
                    "contribution_share": share,
                    "direction": direction,
                }
            )
        )
    return enriched


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _inverse_normalized_mean(values: Iterable[float], *, maximum: float) -> float:
    return _clamp(1.0 - (_mean(values) / maximum))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _risk_level(score: float, thresholds: RiskThresholds) -> str:
    if score >= thresholds.high_risk:
        return "high_risk"
    if score >= thresholds.suspicious:
        return "suspicious"
    return "low_risk"


def _confidence_interval(score: float, sample_size: int) -> tuple[float, float]:
    margin = min(0.25, 1.96 * sqrt(score * (1.0 - score) / sample_size))
    return (round(_clamp(score - margin), 4), round(_clamp(score + margin), 4))
