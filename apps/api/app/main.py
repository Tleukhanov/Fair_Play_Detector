"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router as scans_router
from .config import get_settings
from .db.session import session_factory
from .workers.default_stages import local_scan_stages
from .workers.executor import BackgroundScanExecutor
from .workers.service import ScanWorker
from .workers.store import ScanWorkerStore

settings = get_settings()


def _create_scan_job():
    worker = ScanWorker(ScanWorkerStore(session_factory), local_scan_stages())
    return worker.run

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Own the process-local executor for the lifetime of the API."""

    application.state.scan_executor = BackgroundScanExecutor(
        _create_scan_job(),
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
