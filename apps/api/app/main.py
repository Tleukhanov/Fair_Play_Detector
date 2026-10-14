"""FastAPI application entry point."""

from fastapi import FastAPI

from .config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Probabilistic analysis of public Lichess game behaviour.",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a process-level health response."""

    return {"status": "ok"}
