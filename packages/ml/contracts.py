"""Stable inputs and outputs for model training and scoring workers."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.feature_engineering.schema import GameAggregateFeatures


class MLModel(StrEnum):
    BASELINE = "baseline"
    XGBOOST = "xgboost"


class MLModelContract(BaseModel):
    """Strict JSON-compatible base for ML handoff contracts."""

    model_config = ConfigDict(extra="forbid")


class RiskThresholds(MLModelContract):
    """Calibrated probability thresholds selected on validation data."""

    suspicious: float = Field(ge=0.0, le=1.0)
    high_risk: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def high_risk_above_suspicious(self) -> "RiskThresholds":
        if self.high_risk <= self.suspicious:
            raise ValueError("high_risk threshold must exceed suspicious threshold")
        return self


class ModelMetadata(MLModelContract):
    """Reproducibility metadata persisted with every score."""

    model_version: str = Field(min_length=1)
    model_type: MLModel
    feature_schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    feature_names: list[str] = Field(min_length=1)
    engine_name: str
    engine_version: str
    analysis_depth: int = Field(ge=1)
    analysis_time_ms: int = Field(ge=1)
    trained_at: date | None = None
    dataset_hash: str | None = None
    thresholds: RiskThresholds


class ScoringInput(MLModelContract):
    """Aggregated games supplied to a model; raw PGN is intentionally absent."""

    username_normalized: str
    games: list[GameAggregateFeatures] = Field(min_length=1)


class FeatureEvidence(MLModelContract):
    """One grounded explanation item returned by the numeric scorer."""

    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float
    baseline: float
    weight: float


class ScoreResult(MLModelContract):
    """Calibrated score and explainability data for API/LLM consumers."""

    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    games_analyzed: int = Field(ge=1)
    evidence: list[FeatureEvidence]
    model: ModelMetadata

    @model_validator(mode="after")
    def valid_confidence_interval(self) -> "ScoreResult":
        lower, upper = self.confidence_interval
        if not 0.0 <= lower <= self.risk_score <= upper <= 1.0:
            raise ValueError("confidence interval must contain risk_score within [0, 1]")
        return self
