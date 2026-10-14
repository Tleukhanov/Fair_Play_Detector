"""FastAPI dependencies for worker-owned application services."""

from fastapi import Request

from .executor import BackgroundScanExecutor


def get_scan_executor(request: Request) -> BackgroundScanExecutor:
    """Return the executor created by the application lifespan."""

    return request.app.state.scan_executor
