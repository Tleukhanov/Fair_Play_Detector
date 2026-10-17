"""Bounded process-local background execution for scan jobs."""

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID


class ExecutorCapacityError(RuntimeError):
    """Raised when the bounded executor cannot accept another scan."""


ScanJob = Callable[[UUID], Awaitable[None]]


class BackgroundScanExecutor:
    """Run a bounded number of scans without blocking an HTTP request."""

    def __init__(self, job: ScanJob, *, max_concurrency: int = 2, max_pending: int = 20) -> None:
        if max_concurrency < 1 or max_pending < 0:
            raise ValueError(
                "executor limits must be non-negative and max_concurrency must be positive"
            )
        self._job = job
        self._max_pending = max_pending
        self._pending = 0
        self._concurrency = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def pending_count(self) -> int:
        """Return the number of accepted jobs not yet completed."""

        return len(self._tasks)

    def submit(self, scan_id: UUID) -> None:
        """Accept a job or fail explicitly when the pending limit is reached."""

        if self._pending >= self._max_pending:
            raise ExecutorCapacityError("scan queue capacity exceeded")
        self._pending += 1
        task = asyncio.create_task(self._run(scan_id))
        self._tasks.add(task)
        task.add_done_callback(self._discard_task)

    async def shutdown(self) -> None:
        """Cancel pending work and wait for task cleanup during app shutdown."""

        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(self, scan_id: UUID) -> None:
        async with _SemaphoreContext(self._concurrency):
            await self._job(scan_id)

    def _discard_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._pending -= 1
        if not task.cancelled():
            task.exception()


class _SemaphoreContext:
    """Async context manager for a semaphore without allocating a task."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._semaphore.release()
