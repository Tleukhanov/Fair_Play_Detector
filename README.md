# Fair Play Detector

Fair Play Detector analyzes public Lichess games for statistically unusual,
machine-like patterns. Results are probabilistic signals for additional review,
not proof of cheating or grounds for automatic moderation.

## Development skeleton

The repository contains shared Python contracts, SQLite persistence, and the
initial Lichess streaming ingest package. Workers, ML, LLM, frontend, and
production infrastructure are planned follow-up components.

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn apps.api.app.main:app --reload
```

The health endpoint is available at `GET /health`. API DTOs are defined in
`apps/api/app/schemas/api.py`; the dataset feature contract is defined in
`packages/feature_engineering/schema.py`.

No database or external service is required for the skeleton checks. Keep
credentials in local environment variables and never commit `.env`.

The ingest package exposes `LichessClient` for bounded public exports and
`iter_pgn_games` for per-game PGN parsing. It validates usernames, streams
response chunks, handles transient failures with bounded retry/backoff, and
records malformed-game errors without logging or persisting complete PGNs.
