"""Strict LLM input/output contracts that cannot alter the numeric score."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExplanationModel(BaseModel):
    """Reject unknown fields and keep provider output JSON-compatible."""

    model_config = ConfigDict(extra="forbid")


class ExplanationEvidence(ExplanationModel):
    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float
    baseline: float
    weight: float


class ExplanationInput(ExplanationModel):
    """Only verified score data may be sent to an explanation provider."""

    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    games_analyzed: int = Field(ge=10)
    evidence: list[ExplanationEvidence]
    limitations: list[str] = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def score_in_interval(self) -> "ExplanationInput":
        lower, upper = self.confidence_interval
        if not 0.0 <= lower <= self.risk_score <= upper <= 1.0:
            raise ValueError("confidence interval must contain risk_score")
        return self


class ExplanationOutput(ExplanationModel):
    """Provider response; it contains prose only, never a replacement score."""

    summary: str = Field(min_length=1, max_length=2000)
    limitations: list[str] = Field(min_length=1)
    source: str = Field(pattern=r"^(llm|deterministic)$")
