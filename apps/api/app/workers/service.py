"""Persistence-aware scan worker service."""

from dataclasses import dataclass
from uuid import UUID

from ..domain.models import DomainScanStatus
from .pipeline import ScanPipelineError, ScanStage
from .store import ScanClaimError, ScanWorkerStore


@dataclass(slots=True)
class ScanWorker:
    """Run one claimed scan through injected pipeline stages."""

    store: ScanWorkerStore
    stages: tuple[ScanStage, ...]

    def __post_init__(self) -> None:
        if len(self.stages) != 4:
            raise ValueError("worker requires fetch, analyze, score, and explain stages")

    async def run(self, scan_id: UUID) -> DomainScanStatus:
        """Claim, execute, persist transitions, and return the terminal state."""

        lease_id = await self.store.claim(scan_id)
        current = DomainScanStatus.FETCHING
        targets = (
            DomainScanStatus.ANALYZING,
            DomainScanStatus.SCORING,
            DomainScanStatus.EXPLAINING,
            DomainScanStatus.COMPLETED,
        )
        try:
            for stage, target in zip(self.stages, targets, strict=True):
                await stage.run(scan_id)
                await self.store.transition(scan_id, lease_id, current, target)
                current = target
        except ScanPipelineError as exc:
            await self.store.fail(scan_id, lease_id, exc.code, str(exc))
            return DomainScanStatus.FAILED
        except ScanClaimError:
            raise
        except Exception as exc:
            await self.store.fail(scan_id, lease_id, "WORKER_STAGE_FAILED", str(exc))
            return DomainScanStatus.FAILED
        return current
