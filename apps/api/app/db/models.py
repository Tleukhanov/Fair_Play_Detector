"""SQLAlchemy persistence models matching the shared domain contract."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    lichess_username: Mapped[str] = mapped_column(String(30), nullable=False)
    username_normalized: Mapped[str] = mapped_column(
        String(30), unique=True, index=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    scans: Mapped[list["ScanModel"]] = relationship(back_populates="user")
    games: Mapped[list["GameModel"]] = relationship(back_populates="user")


class ScanModel(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    requested_time_control: Mapped[str] = mapped_column(String(20), nullable=False)
    requested_game_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    games_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    games_analyzed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(100))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    lease_id: Mapped[str | None] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    user: Mapped[UserModel] = relationship(back_populates="scans")
    games: Mapped[list["GameModel"]] = relationship(back_populates="scan")
    verdict: Mapped["VerdictModel | None"] = relationship(back_populates="scan", uselist=False)


class GameModel(Base):
    __tablename__ = "games"
    __table_args__ = (UniqueConstraint("lichess_game_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    lichess_game_id: Mapped[str] = mapped_column(String(20), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), nullable=False, index=True)
    played_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    rated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    speed: Mapped[str] = mapped_column(String(20), nullable=False)
    variant: Mapped[str] = mapped_column(String(30), nullable=False)
    white_username: Mapped[str] = mapped_column(String(30), nullable=False)
    black_username: Mapped[str] = mapped_column(String(30), nullable=False)
    user_color: Mapped[str] = mapped_column(String(5), nullable=False)
    result: Mapped[str] = mapped_column(String(7), nullable=False)
    opening: Mapped[str | None] = mapped_column(String(100))
    pgn: Mapped[str | None] = mapped_column(Text)
    pgn_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    user: Mapped[UserModel] = relationship(back_populates="games")
    scan: Mapped[ScanModel] = relationship(back_populates="games")
    features: Mapped[list["GameFeaturesModel"]] = relationship(back_populates="game")
    scores: Mapped[list["GameScoreModel"]] = relationship(back_populates="game")


class GameFeaturesModel(Base):
    __tablename__ = "game_features"
    __table_args__ = (UniqueConstraint("cache_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    cache_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    pgn_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feature_version: Mapped[str] = mapped_column(String(30), nullable=False)
    aggregates: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    moves: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    engine_name: Mapped[str | None] = mapped_column(String(50))
    engine_version: Mapped[str | None] = mapped_column(String(50))
    depth_or_time_limit: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    game: Mapped[GameModel] = relationship(back_populates="features")


class GameScoreModel(Base):
    __tablename__ = "game_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    top_move_match_rate: Mapped[float] = mapped_column(Float, nullable=False)
    mean_centipawn_loss: Mapped[float] = mapped_column(Float, nullable=False)
    maia_divergence: Mapped[float | None] = mapped_column(Float)
    explainability: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    game: Mapped[GameModel] = relationship(back_populates="scores")


class VerdictModel(Base):
    __tablename__ = "verdicts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), unique=True, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_lower: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_upper: Mapped[float] = mapped_column(Float, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    limitations: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    scan: Mapped[ScanModel] = relationship(back_populates="verdict")
