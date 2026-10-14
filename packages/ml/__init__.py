"""Scoring contracts and baseline model adapter."""

from .baseline import BaselineScorer, InsufficientGamesError
from .contracts import (
    FeatureEvidence,
    ModelMetadata,
    RiskThresholds,
    ScoreResult,
    ScoringInput,
)

__all__ = [
    "BaselineScorer",
    "FeatureEvidence",
    "InsufficientGamesError",
    "ModelMetadata",
    "RiskThresholds",
    "ScoreResult",
    "ScoringInput",
]
