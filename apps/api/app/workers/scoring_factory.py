"""Select the scoring implementation available in the running process.

The API must start before the trained artifact exists.  The factory therefore
imports :mod:`packages.ml.scorer` lazily and falls back to the deterministic
baseline when ``<MODEL_DIR>/metadata.json`` is missing.
"""

from __future__ import annotations

from pathlib import Path

from packages.ml import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    BaselineScorer,
    ModelMetadata,
    RiskThresholds,
)
from packages.ml.contracts import MLModel
from packages.ml.protocols import Scorer

from ..config import Settings

BASELINE_MODEL_VERSION = "baseline-heuristic-v1"


def create_scorer(settings: Settings) -> Scorer:
    """Return a trained scorer when an artifact exists, else the baseline."""

    model_dir = Path(settings.model_dir)
    if (model_dir / "metadata.json").is_file():
        # Imported only when the ML branch has produced an artifact, keeping the
        # API importable while optional ML dependencies are not installed.
        from packages.ml.scorer import TrainedScorer  # type: ignore[import-not-found]

        trained: Scorer = TrainedScorer.load(model_dir)
        return trained
    return BaselineScorer(_baseline_metadata(settings))


def _baseline_metadata(settings: Settings) -> ModelMetadata:
    """Describe the fallback scorer with explicit, reviewable thresholds."""

    return ModelMetadata(
        model_version=BASELINE_MODEL_VERSION,
        model_type=MLModel.BASELINE,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(FEATURE_NAMES),
        engine_name="stockfish",
        # The engine version is only known after the engine is launched.
        engine_version="unknown",
        analysis_depth=settings.analysis_depth,
        analysis_time_ms=settings.analysis_time_ms,
        thresholds=RiskThresholds(suspicious=0.55, high_risk=0.75),
    )


__all__ = ["BASELINE_MODEL_VERSION", "create_scorer"]
