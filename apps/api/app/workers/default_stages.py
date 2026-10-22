"""Scan pipeline stages wired for local execution."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from packages.chess_analysis.protocol import ChessEngine
from packages.ingest import LichessClient, LichessClientError, LichessIngestService
from packages.llm import DeterministicExplainer
from packages.ml.protocols import Scorer

from ..db.models import ScanModel, UserModel
from .analysis_stages import ChessAnalysisStage, ScoringStage
from .pipeline import ScanPipelineError, ScanStage


class LocalNoOpStage:
    """A bounded stage that preserves orchestration without external side effects."""

    async def run(self, _scan_id: UUID) -> None:
        return None


class LichessFetchStage:
    """Fetch and persist the games belonging to one scan."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        ingest_service: LichessIngestService,
    ) -> None:
        self._session_factory = session_factory
        self._ingest_service = ingest_service

    async def run(self, scan_id: UUID) -> None:
        async with self._session_factory() as session:
            scan = await session.scalar(select(ScanModel).where(ScanModel.id == str(scan_id)))
            if scan is None:
                raise ScanPipelineError("SCAN_NOT_FOUND", "scan not found")
            user = await session.scalar(select(UserModel).where(UserModel.id == scan.user_id))
            if user is None:
                raise ScanPipelineError("USER_NOT_FOUND", "scan user not found")
            try:
                await self._ingest_service.ingest_scan(
                    session,
                    scan,
                    username=user.username_normalized,
                    rated_only=scan.rated_only,
                )
            except LichessClientError as exc:
                raise ScanPipelineError("INGEST_FAILED", str(exc)) from exc


def local_scan_stages(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    lichess_client: LichessClient | None = None,
    *,
    engine: ChessEngine | None = None,
    scorer: Scorer | None = None,
    explainer: DeterministicExplainer | None = None,
) -> tuple[ScanStage, ...]:
    """Return the four scan stages: fetch, analyze, score, and explain.

    Analysis and scoring need an engine and a scorer.  When they are not
    supplied the pipeline keeps the previous bounded no-op behaviour, which
    keeps lightweight callers and tests working.
    """

    if session_factory is None:
        return (LocalNoOpStage(), LocalNoOpStage(), LocalNoOpStage(), LocalNoOpStage())
    lichess_client = lichess_client or LichessClient()
    fetch: ScanStage = LichessFetchStage(
        session_factory, LichessIngestService(lichess_client)
    )
    if engine is None or scorer is None:
        return (fetch, LocalNoOpStage(), LocalNoOpStage(), LocalNoOpStage())
    analyze: ScanStage = ChessAnalysisStage(session_factory, engine)
    score: ScanStage = ScoringStage(session_factory, scorer, explainer=explainer)
    # ScoringStage already persisted the verdict; the final stage only carries
    # the scan through EXPLAINING to COMPLETED.
    return (fetch, analyze, score, LocalNoOpStage())
