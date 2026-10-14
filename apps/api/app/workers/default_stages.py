"""Safe local stages used until external integrations are configured."""

from uuid import UUID

from .pipeline import ScanStage


class LocalNoOpStage:
    """A bounded stage that preserves orchestration without external side effects."""

    async def run(self, _scan_id: UUID) -> None:
        return None


def local_scan_stages() -> tuple[ScanStage, ...]:
    """Return fetch, analysis, scoring, and explanation placeholders."""

    return (LocalNoOpStage(), LocalNoOpStage(), LocalNoOpStage(), LocalNoOpStage())
