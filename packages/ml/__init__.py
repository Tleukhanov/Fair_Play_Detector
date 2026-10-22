"""Scoring contracts and baseline model adapter."""

from .baseline import BaselineScorer, InsufficientGamesError
from .contracts import (
    FeatureEvidence,
    ModelMetadata,
    RiskThresholds,
    ScoreResult,
    ScoringInput,
)
from .feature_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FORBIDDEN_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    FeatureContractError,
    Phase,
    validate_feature_columns,
    validate_feature_values,
    validate_player_disjoint,
)
from .protocols import Scorer

__all__ = [
    "BaselineScorer",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FORBIDDEN_COLUMNS",
    "FeatureContractError",
    "FeatureEvidence",
    "GROUP_COLUMN",
    "InsufficientGamesError",
    "ModelMetadata",
    "Phase",
    "RiskThresholds",
    "ScoreResult",
    "Scorer",
    "ScoringInput",
    "TARGET_COLUMN",
    "validate_feature_columns",
    "validate_feature_values",
    "validate_player_disjoint",
]
