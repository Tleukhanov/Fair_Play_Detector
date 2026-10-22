"""Local smoke test: a trained ChessFraud artifact scores real aggregates.

The artifact (``data/models/<version>``) and the ChessFraud parquet tables are
gitignored, so this module skips on a clean checkout and in CI.  It exists to
catch contract drift between the training pipeline and the serve-time scorer:
load the artifact through :class:`TrainedScorer`, feed it aggregates built from
the real dataset, and assert the persisted ``ScoreResult`` contract.

Point ``MODEL_DIR`` at an artifact and ``FAIR_PLAY_DATA_DIR`` at the tables to
run it locally:

    MODEL_DIR=data/models/chessfraud-v1 \
    FAIR_PLAY_DATA_DIR=/path/to/chessfraud pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "data/models/chessfraud-v1"))
DATA_DIR = Path(os.environ.get("FAIR_PLAY_DATA_DIR", "/home/yeraly/Data/fair_play/chessfraud"))
CHESSFRAUD_PARQUET = DATA_DIR / "chessfraud_full.parquet"
MINIMUM_GAMES = 10

pytestmark = [
    pytest.mark.skipif(
        not (MODEL_DIR / "metadata.json").is_file(),
        reason=f"trained artifact metadata is absent: {MODEL_DIR / 'metadata.json'}",
    ),
    pytest.mark.skipif(
        not CHESSFRAUD_PARQUET.is_file(),
        reason=f"ChessFraud parquet is absent: {CHESSFRAUD_PARQUET}",
    ),
]


def test_trained_scorer_returns_a_contract_valid_result() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("joblib")

    from packages.ml.contracts import ScoreResult, ScoringInput
    from packages.ml.data import load_chessfraud
    from packages.ml.game_features import build_game_frame, game_from_record
    from packages.ml.scorer import TrainedScorer

    games = build_game_frame(load_chessfraud(data_dir=DATA_DIR))
    assert len(games) >= MINIMUM_GAMES, "the dataset must expose at least ten games"

    records = games.head(MINIMUM_GAMES).to_dict(orient="records")
    aggregates = [game_from_record(record) for record in records]

    scorer = TrainedScorer.load(MODEL_DIR)
    result = scorer.score(ScoringInput(username_normalized="smoke-user", games=aggregates))

    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.risk_score <= 1.0
    lower, upper = result.confidence_interval
    assert 0.0 <= lower <= result.risk_score <= upper <= 1.0
    assert result.games_analyzed == MINIMUM_GAMES
    assert result.risk_level in {"low_risk", "suspicious", "high_risk"}

    assert result.model.model_version == scorer.artifact_metadata["model_version"]
    assert result.model.feature_names == list(scorer.metadata.feature_names)

    assert result.evidence, "the scorer must return grounded evidence"
    assert all(item.code.startswith("GAME_") for item in result.evidence)
    assert all(math.isfinite(item.weight) for item in result.evidence)
