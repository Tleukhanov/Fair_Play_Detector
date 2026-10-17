"""Explicit scan state transitions for API and worker coordination."""

from collections.abc import Mapping

from ..domain.models import DomainScanStatus


class InvalidScanTransition(ValueError):
    """Raised when a worker attempts an invalid state transition."""


_TRANSITIONS: Mapping[DomainScanStatus, frozenset[DomainScanStatus]] = {
    DomainScanStatus.QUEUED: frozenset({DomainScanStatus.FETCHING, DomainScanStatus.FAILED}),
    DomainScanStatus.FETCHING: frozenset({DomainScanStatus.ANALYZING, DomainScanStatus.FAILED}),
    DomainScanStatus.ANALYZING: frozenset({DomainScanStatus.SCORING, DomainScanStatus.FAILED}),
    DomainScanStatus.SCORING: frozenset({DomainScanStatus.EXPLAINING, DomainScanStatus.FAILED}),
    DomainScanStatus.EXPLAINING: frozenset({DomainScanStatus.COMPLETED, DomainScanStatus.FAILED}),
    DomainScanStatus.COMPLETED: frozenset(),
    DomainScanStatus.FAILED: frozenset(),
}


class ScanStateMachine:
    """Validate the linear pipeline and terminal states."""

    @staticmethod
    def transition(
        current: DomainScanStatus,
        target: DomainScanStatus,
    ) -> DomainScanStatus:
        if target not in _TRANSITIONS[current]:
            raise InvalidScanTransition(f"cannot transition scan from {current} to {target}")
        return target
