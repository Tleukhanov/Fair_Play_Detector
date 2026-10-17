"""Integration checks for the process-local scan execution lifecycle."""

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.api.app.db.base import Base
from apps.api.app.db.models import ScanModel, UserModel
from apps.api.app.workers.default_stages import LichessFetchStage
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
