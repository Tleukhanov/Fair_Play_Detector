# Fair Play Detector

Fair Play Detector analyzes public Lichess games for statistically unusual,
machine-like patterns. Results are probabilistic signals for additional review,
not proof of cheating or grounds for automatic moderation.

## Development skeleton

The repository currently contains the shared Python contracts and API skeleton.
Collectors, workers, persistence, ML, LLM, frontend, and infrastructure are
planned follow-up components.

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
