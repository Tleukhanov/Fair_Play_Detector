"""Stable request and response DTOs for the versioned API."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DTOModel(BaseModel):
    """Base DTO configuration shared by all API models."""

    model_config = ConfigDict(extra="forbid")


class RiskLevel(StrEnum):
    LOW_RISK = "low_risk"
    SUSPICIOUS = "suspicious"
    HIGH_RISK = "high_risk"


class ScanStatus(StrEnum):
    QUEUED = "queued"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    SCORING = "scoring"
    EXPLAINING = "explaining"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanCreateRequest(DTOModel):
    username: str = Field(min_length=1, max_length=30, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
    max_games: int = Field(default=30, ge=10, le=50)
    speed: str = Field(default="blitz", pattern=r"^(blitz|rapid)$")
    rated_only: bool = True


class ScanCreateResponse(DTOModel):
    scan_id: UUID
    status: ScanStatus = ScanStatus.QUEUED
    poll_url: str


class ScanStatusResponse(DTOModel):
    scan_id: UUID
    status: ScanStatus
    games_found: int = Field(default=0, ge=0)
    games_analyzed: int = Field(default=0, ge=0)
    current_stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class EvidenceItem(DTOModel):
    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: float
    baseline: float
    weight: float


class VerdictResponse(DTOModel):
    scan_id: UUID
    username: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    games_analyzed: int = Field(ge=10)
    summary: str
    evidence: list[EvidenceItem]
    limitations: list[str] = Field(min_length=1)
    model_version: str


class GameAggregateDTO(DTOModel):
    game_id: str
    source_url: HttpUrl
    played_at: datetime
    speed: str
    result: str
    user_color: str
    move_count: int = Field(ge=0)
    top_move_match_rate: float = Field(ge=0.0, le=1.0)
    mean_centipawn_loss: float = Field(ge=0.0)
    maia_divergence: float | None = Field(default=None, ge=0.0)


class GamesResponse(DTOModel):
    scan_id: UUID
    items: list[GameAggregateDTO]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


class GameMoveDTO(DTOModel):
    ply: int = Field(ge=1)
    fen_before: str
    played_move_uci: str
    played_move_san: str | None = None
    centipawn_loss: float = Field(ge=0.0)
    is_top_1: bool
    is_top_3: bool
    time_seconds: float | None = Field(default=None, ge=0.0)
    maia_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class GameDetailResponse(DTOModel):
    scan_id: UUID
    game: GameAggregateDTO
    moves: list[GameMoveDTO]
