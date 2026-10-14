"""Versioned, serializable feature contract shared by analysis and ML agents."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FeatureModel(BaseModel):
    """Base configuration for rows written to Parquet or exchanged as JSON."""

    model_config = ConfigDict(extra="forbid")


class DatasetClass(StrEnum):
    HUMAN = "human"
    ENGINE_LIKE = "engine_like"


class MoveFeature(FeatureModel):
    """One analyzed move by the requested player."""

    ply: int = Field(ge=1)
    fen_before: str
    played_move_uci: str
    played_move_san: str
    eval_before_cp: int
    eval_best_move_cp: int
    eval_played_move_cp: int
    centipawn_loss: float = Field(ge=0.0)
    is_top_1: bool
    is_top_3: bool
    time_seconds: float | None = Field(default=None, ge=0.0)
    analysis_depth: int | None = Field(default=None, ge=1)
    maia_eval: float | None = None
    maia_move_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    is_forced: bool = False
    is_noisy_position: bool = False


class GameAggregateFeatures(FeatureModel):
    """Aggregates for one game; names are stable model input columns."""

    move_count: int = Field(ge=1)
    mean_centipawn_loss: float = Field(ge=0.0)
    median_centipawn_loss: float = Field(ge=0.0)
    p90_centipawn_loss: float = Field(ge=0.0)
    top_1_match_rate: float = Field(ge=0.0, le=1.0)
    top_3_match_rate: float = Field(ge=0.0, le=1.0)
    difficult_position_match_rate: float = Field(ge=0.0, le=1.0)
    tactical_error_count: int = Field(ge=0)
    centipawn_loss_stability: float = Field(ge=0.0)
    identical_time_share: float = Field(ge=0.0, le=1.0)
    median_time_seconds: float | None = Field(default=None, ge=0.0)
    mean_time_seconds: float | None = Field(default=None, ge=0.0)
    time_variance: float | None = Field(default=None, ge=0.0)
    time_difficulty_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    maia_divergence: float | None = Field(default=None, ge=0.0)


class FeatureRecord(FeatureModel):
    """One dataset row and its provenance metadata."""

    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dataset_class: DatasetClass
    game_id: str
    username_normalized: str
    speed: str
    rated: bool
    result: str
    user_color: str
    aggregate: GameAggregateFeatures
    moves: list[MoveFeature] = Field(min_length=1)
    engine_name: str
    engine_version: str
    analysis_config: str
