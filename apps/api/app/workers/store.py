"""Persistence operations used by scan workers."""

from datetime import datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Result, func, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..db.models import ScanModel
from ..domain.models import DomainScanStatus
from .state_machine import ScanStateMachine

_TERMINAL_STATUSES = (
    DomainScanStatus.COMPLETED.value,
    DomainScanStatus.FAILED.value,
)


class ScanClaimError(RuntimeError):
    """Raised when a scan cannot be claimed or updated by a worker."""


def _rowcount(result: Result[Any]) -> int:
    """Return the affected-row count of a DML statement."""

    return int(cast(CursorResult[Any], result).rowcount)


class ScanWorkerStore:
    """Apply worker-owned state changes with lease protection."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def claim(self, scan_id: UUID) -> str:
        """Atomically claim a queued scan and move it to fetching."""

        lease_id = str(uuid4())
        async with self._session_factory() as session:
            result = await session.execute(
                update(ScanModel)
                .where(
                    ScanModel.id == str(scan_id),
                    ScanModel.status == DomainScanStatus.QUEUED.value,
                    ScanModel.lease_id.is_(None),
                )
                .values(
                    status=DomainScanStatus.FETCHING.value,
                    lease_id=lease_id,
                    started_at=datetime.utcnow(),
                )
            )
            if _rowcount(result) != 1:
                await session.rollback()
                raise ScanClaimError("scan is missing, already claimed, or not queued")
            await session.commit()
        return lease_id

    async def transition(
        self,
        scan_id: UUID,
        lease_id: str,
        current: DomainScanStatus,
        target: DomainScanStatus,
    ) -> None:
        """Move a leased scan to the next valid state."""

        ScanStateMachine.transition(current, target)
        values: dict[str, Any] = {"status": target.value}
        if target is DomainScanStatus.COMPLETED:
            values["completed_at"] = datetime.utcnow()
            values["error_code"] = None
            values["error_message"] = None
        async with self._session_factory() as session:
            result = await session.execute(
                update(ScanModel)
                .where(
                    ScanModel.id == str(scan_id),
                    ScanModel.lease_id == lease_id,
                    ScanModel.status == current.value,
                )
                .values(**values)
            )
            if _rowcount(result) != 1:
                await session.rollback()
                raise ScanClaimError("scan lease is invalid or state changed")
            await session.commit()

    async def fail(self, scan_id: UUID, lease_id: str, code: str, message: str) -> None:
        """Mark a leased scan failed without exposing internal exception details."""

        async with self._session_factory() as session:
            result = await session.execute(
                update(ScanModel)
                .where(
                    ScanModel.id == str(scan_id),
                    ScanModel.lease_id == lease_id,
                    ScanModel.status.not_in(_TERMINAL_STATUSES),
                )
                .values(
                    status=DomainScanStatus.FAILED.value,
                    error_code=code,
                    error_message=message,
                    completed_at=datetime.utcnow(),
                )
            )
            if _rowcount(result) != 1:
                await session.rollback()
                raise ScanClaimError("scan lease is invalid or scan is terminal")
            await session.commit()


async def reap_stale_scans(
    session_factory: async_sessionmaker[AsyncSession],
    older_than_minutes: int,
) -> int:
    """Fail non-terminal scans idle for longer than ``older_than_minutes``.

    Called on process start so a crashed worker never leaves a scan polling
    forever.  Returns the number of scans that were marked failed.
    """

    if older_than_minutes < 1:
        raise ValueError("older_than_minutes must be positive")
    now = datetime.utcnow()
    threshold = now - timedelta(minutes=older_than_minutes)
    async with session_factory() as session:
        result = await session.execute(
            update(ScanModel)
            .where(
                ScanModel.status.not_in(_TERMINAL_STATUSES),
                func.coalesce(ScanModel.started_at, ScanModel.created_at) < threshold,
            )
            .values(
                status=DomainScanStatus.FAILED.value,
                error_code="STALE_SCAN",
                error_message="scan exceeded the worker lease window",
                completed_at=now,
            )
        )
        await session.commit()
        return _rowcount(result)
