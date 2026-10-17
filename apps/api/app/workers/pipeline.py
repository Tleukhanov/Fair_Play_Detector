"""Worker pipeline protocol without a concrete queue implementation."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ..domain.models import DomainScanStatus
from .state_machine import ScanStateMachine


class ScanPipelineError(RuntimeError):
    """Raised when a pipeline stage fails with an explicit error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ScanStage(Protocol):
    """One stage that performs bounded work for a scan."""

    async def run(self, scan_id: UUID) -> None: ...


@dataclass(slots=True)
class ScanPipeline:
    """Orchestrate stage ordering; persistence/queue adapters are injected later."""

    stages: tuple[ScanStage, ...]
    state_machine: ScanStateMachine = ScanStateMachine()

    def __post_init__(self) -> None:
        if len(self.stages) != 4:
            raise ValueError("pipeline requires fetch, analyze, score, and explain stages")

    async def run(self, scan_id: UUID) -> DomainScanStatus:
        """Run configured stages in order and return the terminal status."""

        status = DomainScanStatus.QUEUED
        targets = (
            DomainScanStatus.FETCHING,
            DomainScanStatus.ANALYZING,
            DomainScanStatus.SCORING,
            DomainScanStatus.EXPLAINING,
        )
        for stage, target in zip(self.stages, targets, strict=True):
            status = self.state_machine.transition(status, target)
            try:
                await stage.run(scan_id)
            except ScanPipelineError:
                self.state_machine.transition(status, DomainScanStatus.FAILED)
                return DomainScanStatus.FAILED
        return self.state_machine.transition(status, DomainScanStatus.COMPLETED)
