"""Strict LLM input/output contracts that cannot alter the numeric score.

``ExplanationInput`` is the only payload that may ever be sent to a language
model.  It deliberately has **no identity fields**: no username, no PGN, no game
IDs, no player or opponent names, and no ratings.  It carries the already
computed numeric verdict plus grounded evidence only.  Keeping identity out of
``ExplanationInput`` is a privacy contract, not an implementation detail, so the
field list must not grow an identity field later.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExplanationModel(BaseModel):
    """Reject unknown fields and keep provider output JSON-compatible."""

    model_config = ConfigDict(extra="forbid")


class ExplanationDirection(StrEnum):
    """Whether a feature pushed the model output towards or away from risk."""

    SUPPORTS_RISK = "supports_risk"
    REDUCES_RISK = "reduces_risk"
    NEUTRAL = "neutral"


class ExplanationEvidence(ExplanationModel):
    """One numeric evidence item, optionally with a per-verdict contribution.

    ``contribution`` is the model-space attribution for this verdict (XGBoost
    margin-space SHAP value for the trained model, or a weighted delta from the
    evidence baseline for the heuristic scorer).  It is optional so existing
    callers that only know the global weight keep working.  ``contribution_share``
    is the share of total absolute attribution and ``direction`` is derived from
    the contribution sign.
    """

    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float
    baseline: float
    weight: float
    contribution: float | None = None
    contribution_share: float | None = None
    direction: ExplanationDirection | None = None


class ExplanationThresholds(ExplanationModel):
    """Player-level decision thresholds on the calibrated risk scale."""

    suspicious: float = Field(ge=0.0, le=1.0)
    high_risk: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def high_risk_above_suspicious(self) -> "ExplanationThresholds":
        if self.high_risk <= self.suspicious:
            raise ValueError("high_risk threshold must exceed suspicious threshold")
        return self


class ExplanationInput(ExplanationModel):
    """Only verified score data may be sent to an explanation provider.

    Identity is intentionally absent: never add a username, PGN, game ID, player
    ID, opponent name, or rating to this model.  The explainer only needs the
    numeric verdict and grounded feature evidence.
    """

    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    games_analyzed: int = Field(ge=10)
    evidence: list[ExplanationEvidence]
    limitations: list[str] = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    thresholds: ExplanationThresholds | None = None

    @model_validator(mode="after")
    def score_in_interval(self) -> "ExplanationInput":
        lower, upper = self.confidence_interval
        if not 0.0 <= lower <= self.risk_score <= upper <= 1.0:
            raise ValueError("confidence interval must contain risk_score")
        return self


class ExplanationReasoning(ExplanationModel):
    """One causal-in-model bullet tied to a concrete evidence code."""

    evidence_code: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    direction: ExplanationDirection
    contribution: float | None = None


class ExplanationSource(ExplanationModel):
    """A retrieved knowledge chunk cited by the explanation."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)


class ExplanationOutput(ExplanationModel):
    """Provider response; it contains prose only, never a replacement score."""

    summary: str = Field(min_length=1, max_length=2000)
    limitations: list[str] = Field(min_length=1)
    source: str = Field(pattern=r"^(llm|deterministic)$")
    reasoning: list[ExplanationReasoning] = Field(default_factory=list)
    sources: list[ExplanationSource] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    prompt_version: str = Field(default="deterministic-v1", min_length=1)
    knowledge_version: str = Field(default="1.0.0", min_length=1)
    fallback_reason: str | None = None


__all__ = [
    "ExplanationDirection",
    "ExplanationEvidence",
    "ExplanationInput",
    "ExplanationModel",
    "ExplanationOutput",
    "ExplanationReasoning",
    "ExplanationSource",
    "ExplanationThresholds",
]
