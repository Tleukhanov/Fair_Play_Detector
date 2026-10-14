"""Persistence-neutral domain entities from the project plan."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class DomainRiskLevel(StrEnum):
    LOW_RISK = "low_risk"
    SUSPICIOUS = "suspicious"
    HIGH_RISK = "high_risk"


class DomainScanStatus(StrEnum):
    QUEUED = "queued"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    SCORING = "scoring"
    EXPLAINING = "explaining"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class User:
    id: UUID
    lichess_username: str
    username_normalized: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Scan:
    id: UUID
    user_id: UUID
    status: DomainScanStatus
    requested_time_control: str
    requested_game_limit: int
    games_found: int = 0
    games_analyzed: int = 0
    model_version: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Game:
    id: UUID
    lichess_game_id: str
    user_id: UUID
    scan_id: UUID
    played_at: datetime
    rated: bool
    speed: str
    variant: str
    white_username: str
    black_username: str
    user_color: str
    result: str
    source_url: str
    pgn_hash: str
    pgn: str | None = None


@dataclass(frozen=True, slots=True)
class GameFeatures:
    id: UUID
    game_id: UUID
    feature_version: str
    aggregates: dict[str, float | int | None]
    moves: list[dict[str, Any]] = field(default_factory=list)
    engine_name: str | None = None
    engine_version: str | None = None
    depth_or_time_limit: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class GameScore:
    id: UUID
    game_id: UUID
    model_version: str
    risk_score: float
    top_move_match_rate: float
    mean_centipawn_loss: float
    maia_divergence: float | None
    explainability: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Verdict:
    id: UUID
    scan_id: UUID
    risk_level: DomainRiskLevel
    risk_score: float
    summary: str
    evidence: list[dict[str, Any]]
    limitations: list[str]
    model_version: str
    prompt_version: str | None = None
    created_at: datetime | None = None
