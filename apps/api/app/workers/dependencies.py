"""FastAPI dependencies for worker-owned application services."""

from typing import cast

from fastapi import Request

from .executor import BackgroundScanExecutor


def get_scan_executor(request: Request) -> BackgroundScanExecutor:
    """Return the executor created by the application lifespan."""

    return cast(BackgroundScanExecutor, request.app.state.scan_executor)
