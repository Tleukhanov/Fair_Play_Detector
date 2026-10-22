"""Integration checks for the process-local scan execution lifecycle."""

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.api.app.db.base import Base
from apps.api.app.db.models import ScanModel, UserModel
from apps.api.app.workers.analysis_stages import ChessAnalysisStage, ScoringStage
from apps.api.app.workers.default_stages import (
    LichessFetchStage,
    LocalNoOpStage,
    local_scan_stages,
)
from apps.api.app.workers.executor import BackgroundScanExecutor, ExecutorCapacityError
from apps.api.app.workers.pipeline import ScanPipelineError


def test_executor_runs_scan_and_reclaims_capacity() -> None:
    seen: list[str] = []

    async def job(scan_id) -> None:
        seen.append(str(scan_id))

    async def scenario() -> None:
        executor = BackgroundScanExecutor(job, max_concurrency=1, max_pending=1)
        scan_id = uuid4()
        executor.submit(scan_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert seen == [str(scan_id)]
        assert executor.pending_count == 0
        await executor.shutdown()

    asyncio.run(scenario())


def test_executor_rejects_over_capacity_and_shutdown_is_graceful() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def job(_scan_id) -> None:
        started.set()
        await release.wait()

    async def scenario() -> None:
        executor = BackgroundScanExecutor(job, max_concurrency=1, max_pending=1)
        executor.submit(uuid4())
        await started.wait()
        with pytest.raises(ExecutorCapacityError):
            executor.submit(uuid4())
        await executor.shutdown()
        release.set()
        assert executor.pending_count == 0

    asyncio.run(scenario())


def test_lichess_fetch_stage_uses_persisted_scan_user() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        user = UserModel(
            id=str(uuid4()),
            lichess_username="Tester",
            username_normalized="tester",
        )
        scan = ScanModel(
            id=str(uuid4()),
            user=user,
            status="fetching",
            requested_time_control="blitz",
            requested_game_limit=10,
        )
        async with session_factory() as session:
            session.add(scan)
            await session.commit()

        calls: list[tuple[str, str]] = []

        class FakeIngestService:
            async def ingest_scan(self, session, received_scan, *, username, rated_only=True):
                calls.append((received_scan.id, username))

        with pytest.raises(ScanPipelineError, match="scan not found"):
            await LichessFetchStage(session_factory, FakeIngestService()).run(uuid4())
        assert calls == []
        await LichessFetchStage(session_factory, FakeIngestService()).run(
            UUID(scan.id)
        )
        assert calls == [(scan.id, "tester")]
        await engine.dispose()

    asyncio.run(scenario())


class _FakeIngestService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    async def ingest_scan(self, session, received_scan, *, username, rated_only=True):
        self.calls.append((received_scan.id, username, rated_only))


def test_lichess_fetch_stage_forwards_rated_only() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        user = UserModel(
            id=str(uuid4()),
            lichess_username="Tester",
            username_normalized="tester",
        )
        scan = ScanModel(
            id=str(uuid4()),
            user=user,
            status="fetching",
            requested_time_control="blitz",
            requested_game_limit=10,
            rated_only=False,
        )
        async with session_factory() as session:
            session.add(scan)
            await session.commit()

        service = _FakeIngestService()
        await LichessFetchStage(session_factory, service).run(UUID(scan.id))
        assert service.calls == [(scan.id, "tester", False)]
        await engine.dispose()

    asyncio.run(scenario())


class _FakeEngine:
    @property
    def config(self):
        from packages.chess_analysis.models import AnalysisConfig

        return AnalysisConfig(
            engine_name="stockfish", engine_version="fixture", depth=15, time_ms=100
        )

    async def analyze_game(self, pgn, username):
        return []


class _FakeScorer:
    def score(self, scoring_input):
        raise AssertionError("not executed in this test")


def test_local_scan_stages_wires_the_four_stages() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        stages = local_scan_stages(
            session_factory,
            engine=_FakeEngine(),
            scorer=_FakeScorer(),
        )
        assert [type(stage) for stage in stages] == [
            LichessFetchStage,
            ChessAnalysisStage,
            ScoringStage,
            LocalNoOpStage,
        ]
        placeholders = local_scan_stages(session_factory, engine=None, scorer=None)
        assert [type(stage) for stage in placeholders] == [
            LichessFetchStage,
            LocalNoOpStage,
            LocalNoOpStage,
            LocalNoOpStage,
        ]
        assert len(local_scan_stages()) == 4
        await engine.dispose()

    asyncio.run(scenario())
