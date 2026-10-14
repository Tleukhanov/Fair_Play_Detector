"""Persistence operations used by scan workers."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..db.models import ScanModel
from ..domain.models import DomainScanStatus
from .state_machine import ScanStateMachine


class ScanClaimError(RuntimeError):
    """Raised when a scan cannot be claimed or updated by a worker."""


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
            if result.rowcount != 1:
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
        async with self._session_factory() as session:
            result = await session.execute(
                update(ScanModel)
                .where(
                    ScanModel.id == str(scan_id),
                    ScanModel.lease_id == lease_id,
                    ScanModel.status == current.value,
                )
                .values(status=target.value)
            )
            if result.rowcount != 1:
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
                    ScanModel.status.not_in(
                        [DomainScanStatus.COMPLETED.value, DomainScanStatus.FAILED.value]
                    ),
                )
                .values(
                    status=DomainScanStatus.FAILED.value,
                    error_code=code,
                    error_message=message,
                    completed_at=datetime.utcnow(),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                raise ScanClaimError("scan lease is invalid or scan is terminal")
            await session.commit()
