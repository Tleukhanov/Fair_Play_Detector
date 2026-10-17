"""Serializable dataset feature and collection-report contracts."""

from pydantic import BaseModel, ConfigDict, Field


class DatasetFeatureRow(BaseModel):
    """Feature-only dataset row; raw PGN is intentionally not represented."""

    model_config = ConfigDict(extra="forbid")

    dataset_class: str = Field(pattern=r"^(human|engine_like)$")
    game_id: str = Field(min_length=1)
    username: str | None = None
    move_count: int = Field(ge=1)
    mean_cpl: float | None = Field(default=None, ge=0.0)
    top1_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    top3_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_move_time_ms: float | None = Field(default=None, ge=0.0)
    rated: bool | None = None
    speed: str | None = None
    result: str | None = None


class SourceCollectionReport(BaseModel):
    """Counters and failure reasons for one bounded source."""

    source: str
    period: str
    url: str
    downloaded_bytes: int = Field(ge=0)
    games_read: int = Field(ge=0)
    games_processed: int = Field(ge=0)
    games_skipped: int = Field(ge=0)
    skip_reasons: dict[str, int]


class DatasetCollectionReport(BaseModel):
    """Gate artifact written only after both Parquet files validate."""

    report_version: str = "1.0.0"
    feature_schema_version: str = "1.0.0"
    lichess_elite_month: str
    tcec_season: str
    quota_bytes: int
    downloaded_bytes: int
    parquet_files: dict[str, dict[str, object]]
    sources: list[SourceCollectionReport]
    class_counts: dict[str, int]
    class_percentages: dict[str, float]
    sample_rows: list[DatasetFeatureRow] = Field(min_length=5, max_length=5)
    columns: list[str]
    training_started: bool = False
