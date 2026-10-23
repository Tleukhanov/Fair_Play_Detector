"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from packages.chess_analysis import StockfishEngine
from packages.ingest import LichessClient
from packages.llm import DeterministicExplainer

from .api.routes import router as scans_router
from .config import get_settings
from .db.session import session_factory
from .workers.default_stages import local_scan_stages
from .workers.executor import BackgroundScanExecutor, ScanJob
from .workers.scoring_factory import create_scorer
from .workers.service import ScanWorker
from .workers.store import ScanWorkerStore, reap_stale_scans

logger = logging.getLogger(__name__)
settings = get_settings()

#: Static no-build frontend, served from the same origin as the API.
WEB_DIR = Path(__file__).resolve().parents[3] / "apps" / "web"


def _create_scan_job() -> ScanJob:
    """Build the process-local job that runs one scan end to end."""

    engine = StockfishEngine(
        settings.stockfish_path,
        depth=settings.analysis_depth,
        time_ms=settings.analysis_time_ms,
    )
    worker = ScanWorker(
        ScanWorkerStore(session_factory),
        local_scan_stages(
            session_factory,
            LichessClient(str(settings.lichess_api_base_url)),
            engine=engine,
            scorer=create_scorer(settings),
            explainer=DeterministicExplainer(),
        ),
    )
    return worker.run


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Own the process-local executor for the lifetime of the API."""

    try:
        reaped = await reap_stale_scans(session_factory, settings.stale_scan_minutes)
    except Exception as exc:
        # A missing/unmigrated database must not prevent the API from starting;
        # the reaper is best-effort housekeeping.
        logger.warning("stale scan reaper failed during startup: %s", exc)
    else:
        if reaped:
            logger.warning("marked %d stale scans as failed", reaped)
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
app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    """Send the bare origin to the static frontend shell."""

    return RedirectResponse("/app/")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a process-level health response."""

    return {"status": "ok"}
