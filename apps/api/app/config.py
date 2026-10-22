"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Debian/Ubuntu install Stockfish outside PATH; prefer it when present.
_DEFAULT_STOCKFISH_PATH = (
    "/usr/games/stockfish" if Path("/usr/games/stockfish").exists() else "stockfish"
)


class Settings(BaseSettings):
    """Runtime settings shared by API and future workers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="development", validation_alias="APP_ENV")
    app_name: str = Field(default="fair-play-detector", validation_alias="APP_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, validation_alias="API_PORT")
    lichess_api_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://lichess.org"),
        validation_alias="LICHESS_API_BASE_URL",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/fair_play.db",
        validation_alias="DATABASE_URL",
    )
    redis_url: str = Field(
        default="",
        validation_alias="REDIS_URL",
    )
    analysis_depth: int = Field(default=15, ge=1, validation_alias="ANALYSIS_DEPTH")
    analysis_time_ms: int = Field(default=100, ge=1, validation_alias="ANALYSIS_TIME_MS")
    stockfish_path: str = Field(
        default=_DEFAULT_STOCKFISH_PATH, validation_alias="STOCKFISH_PATH"
    )
    model_dir: str = Field(default="data/models/chessfraud-v1", validation_alias="MODEL_DIR")
    stale_scan_minutes: int = Field(default=60, ge=1, validation_alias="STALE_SCAN_MINUTES")
    max_concurrent_scans: int = Field(default=2, ge=1, validation_alias="MAX_CONCURRENT_SCANS")
    max_pending_scans: int = Field(default=20, ge=0, validation_alias="MAX_PENDING_SCANS")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings instance."""

    return Settings()
