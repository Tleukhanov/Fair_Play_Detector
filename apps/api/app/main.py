"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Fair Play Detector API",
    version="0.1.0",
    description="Probabilistic analysis of public Lichess game behaviour.",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a process-level health response."""

    return {"status": "ok"}
