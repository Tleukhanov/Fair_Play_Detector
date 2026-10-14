"""Integration checks for the process-local scan execution lifecycle."""

import asyncio
from uuid import uuid4

import pytest

from apps.api.app.workers.executor import BackgroundScanExecutor, ExecutorCapacityError


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
