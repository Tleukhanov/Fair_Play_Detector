"""Serve a calibrated game-level ChessFraud model through the Scorer protocol.

The artifact written by :mod:`packages.ml.train` contains ``model.joblib``
(calibrated estimator) and ``metadata.json`` (feature contract, thresholds,
engine provenance, global importance, and training baselines).  This module
loads both, predicts one probability per game, and aggregates the games of a
player into a single risk score.

Aggregation is the **mean of game probabilities**, not the maximum: a single
unusual game should not dominate a verdict, and the mean keeps the score
interpretable as "expected share of flagged games".  The confidence interval is
a deterministic percentile bootstrap over games (the only sampling unit
available at serve time); it is not a substitute for the player-level bootstrap
used during evaluation.

Explanations are strictly grounded: evidence values are the player's mean of a
feature, baselines are the training means, and weights are the artifact's global
feature importances.  No evidence item is invented for a feature that the model
did not see.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .baseline import BaselineScorer, InsufficientGamesError
from .contracts import (
    FeatureEvidence,
    ModelMetadata,
    ScoreResult,
    ScoringInput,
)
from .game_features import (
    GAME_FEATURE_NAMES,
    assert_game_feature_contract,
    game_feature_values,
    game_feature_vector,
)

if TYPE_CHECKING:
    from packages.feature_engineering.schema import GameAggregateFeatures

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

#: Human labels for the known game features; unknown names fall back to the code.
_EVIDENCE_LABELS: dict[str, str] = {
    "mean_centipawn_loss": "Mean centipawn loss",
    "median_centipawn_loss": "Median centipawn loss",
    "p90_centipawn_loss": "90th percentile centipawn loss",
    "top_1_match_rate": "Share of moves matching the engine top move",
    "tactical_error_count": "Count of large centipawn losses",
    "centipawn_loss_stability": "Variance of centipawn loss",
    "identical_time_share": "Share of moves with identical thinking time",
    "median_time_seconds": "Median thinking time",
    "mean_time_seconds": "Mean thinking time",
    "time_variance": "Variance of thinking time",
    "has_timing": "Clock annotations available",
}

_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "model_version",
    "model_type",
    "feature_schema_version",
    "feature_names",
    "engine_name",
    "engine_version",
    "analysis_depth",
    "analysis_time_ms",
    "trained_at",
    "dataset_hash",
    "thresholds",
)


class TrainedScorer:
    """Calibrated game-level model adapter implementing :class:`Scorer`."""

    #: Same policy as the heuristic baseline: fewer games cannot be scored.
    minimum_games: int = BaselineScorer.minimum_games
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 2025
    evidence_limit: int = 5

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        *,
        feature_importances: Sequence[tuple[str, float]],
        feature_baselines: dict[str, float],
        artifact_metadata: dict[str, Any],
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._feature_importances = list(feature_importances)
        self._feature_baselines = dict(feature_baselines)
        self._artifact_metadata = dict(artifact_metadata)

    @classmethod
    def load(cls, model_dir: str | Path) -> TrainedScorer:
        """Load ``model.joblib`` and ``metadata.json`` from ``model_dir``."""

        import joblib  # type: ignore[import-untyped]

        directory = Path(model_dir)
        model_path = directory / MODEL_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"trained model artifact not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        missing = [key for key in _REQUIRED_METADATA_KEYS if key not in payload]
        if missing:
            raise ValueError(f"{metadata_path} is missing metadata keys: {missing}")
        metadata = ModelMetadata.model_validate(
            {key: payload[key] for key in _REQUIRED_METADATA_KEYS}
        )
        assert_game_feature_contract(metadata.feature_names)

        raw_importances = payload.get("global_feature_importance")
        if not isinstance(raw_importances, list) or not raw_importances:
            raise ValueError(
                f"{metadata_path} has no global_feature_importance; "
                "cannot produce grounded evidence"
            )
        importances: list[tuple[str, float]] = []
        for item in raw_importances:
            if not isinstance(item, dict):
                raise ValueError(f"invalid feature importance entry: {item!r}")
            name = item.get("name")
            value = item.get("importance")
            if name not in GAME_FEATURE_NAMES:
                raise ValueError(f"importance refers to unknown feature: {name!r}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"importance for {name!r} is not numeric: {value!r}")
            importances.append((str(name), float(value)))

        raw_baselines = payload.get("feature_baselines")
        if not isinstance(raw_baselines, dict):
            raise ValueError(f"{metadata_path} has no feature_baselines mapping")
        baselines: dict[str, float] = {}
        for name in GAME_FEATURE_NAMES:
            value = raw_baselines.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"missing numeric feature baseline for {name!r}")
            baselines[name] = float(value)

        model = joblib.load(model_path)
        return cls(
            model,
            metadata,
            feature_importances=importances,
            feature_baselines=baselines,
            artifact_metadata=payload,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        return dict(self._artifact_metadata)

    def score(self, scoring_input: ScoringInput) -> ScoreResult:
        """Score one player from their aggregated games.

        Raises :class:`InsufficientGamesError` when fewer than
        :attr:`minimum_games` games carry a finite feature vector.
        """

        usable_games: list[GameAggregateFeatures] = []
        vectors: list[list[float]] = []
        for game in scoring_input.games:
            try:
                vector = game_feature_vector(game)
            except (ValueError, TypeError):
                continue
            if all(isfinite(value) for value in vector):
                usable_games.append(game)
                vectors.append(vector)
        if len(vectors) < self.minimum_games:
            raise InsufficientGamesError(
                f"at least {self.minimum_games} usable games are required for scoring; "
                f"got {len(vectors)} from {len(scoring_input.games)} supplied"
            )

        probabilities = self._predict_probabilities(vectors)
        risk_score = _clamp(sum(probabilities) / len(probabilities))
        lower, upper = self._bootstrap_interval(probabilities)
        lower = _clamp(min(lower, risk_score))
        upper = _clamp(max(upper, risk_score))

        return ScoreResult(
            risk_level=_risk_level(risk_score, self._metadata),
            risk_score=risk_score,
            confidence_interval=(lower, upper),
            games_analyzed=len(usable_games),
            evidence=self._evidence(usable_games),
            model=self._metadata,
        )

    def _predict_probabilities(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        raw = self._model.predict_proba([list(vector) for vector in vectors])
        probabilities = [float(row[1]) for row in raw]
        if len(probabilities) != len(vectors):
            raise ValueError("model returned an unexpected number of probabilities")
        return probabilities

    def _bootstrap_interval(self, probabilities: Sequence[float]) -> tuple[float, float]:
        """Percentile bootstrap of the mean over games, deterministic seed."""

        values = list(probabilities)
        if len(values) < 2:
            value = _clamp(values[0]) if values else 0.0
            return (value, value)
        rng = random.Random(self.bootstrap_seed)
        size = len(values)
        means: list[float] = []
        for _ in range(self.bootstrap_iterations):
            total = 0.0
            for _ in range(size):
                total += values[rng.randrange(size)]
            means.append(total / size)
        means.sort()
        return (_percentile(means, 0.025), _percentile(means, 0.975))

    def _evidence(self, games: Sequence[GameAggregateFeatures]) -> list[FeatureEvidence]:
        values_by_game = [game_feature_values(game) for game in games]
        means: dict[str, float] = {}
        for name in GAME_FEATURE_NAMES:
            values = [row[name] for row in values_by_game]
            means[name] = sum(values) / len(values)
        evidence = []
        for name, weight in self._feature_importances[: self.evidence_limit]:
            evidence.append(
                FeatureEvidence(
                    code=f"GAME_{name.upper()}",
                    label=_EVIDENCE_LABELS.get(name, name.replace("_", " ")),
                    value=float(means[name]),
                    baseline=float(self._feature_baselines[name]),
                    weight=float(weight),
                )
            )
        return evidence


def _risk_level(score: float, metadata: ModelMetadata) -> str:
    if score >= metadata.thresholds.high_risk:
        return "high_risk"
    if score >= metadata.thresholds.suspicious:
        return "suspicious"
    return "low_risk"


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(len(ordered) - 1, lower_index + 1)
    fraction = position - lower_index
    return float(ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction)


__all__: Sequence[str] = ("TrainedScorer",)
