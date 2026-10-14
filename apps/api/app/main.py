"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router as scans_router
from .config import get_settings
from .workers.executor import BackgroundScanExecutor

settings = get_settings()


async def _unconfigured_scan_job(_scan_id: object) -> None:
    """Placeholder until the concrete scan worker is wired in."""


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Own the process-local executor for the lifetime of the API."""

    application.state.scan_executor = BackgroundScanExecutor(
        _unconfigured_scan_job,
        max_concurrency=settings.max_concurrent_scans,
        max_pending=settings.max_pending_scans,
    )
    yield
    await application.state.scan_executor.shutdown()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Probabilistic analysis of public Lichess game behaviour.",
    lifespan=lifespan,
)
app.include_router(scans_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a process-level health response."""

    return {"status": "ok"}
