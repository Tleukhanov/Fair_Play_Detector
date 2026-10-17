"""Background scan orchestration contracts."""

from .default_stages import LocalNoOpStage, local_scan_stages
from .executor import BackgroundScanExecutor, ExecutorCapacityError
from .pipeline import ScanPipeline, ScanPipelineError
from .service import ScanWorker
from .state_machine import InvalidScanTransition, ScanStateMachine
from .store import ScanClaimError, ScanWorkerStore

__all__ = [
    "InvalidScanTransition",
    "BackgroundScanExecutor",
    "ExecutorCapacityError",
    "LocalNoOpStage",
    "local_scan_stages",
    "ScanPipeline",
    "ScanPipelineError",
    "ScanStateMachine",
    "ScanClaimError",
    "ScanWorkerStore",
    "ScanWorker",
]
