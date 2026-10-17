"""Regression tests for worker state and scoring contracts."""

import pytest

from apps.api.app.domain.models import DomainScanStatus
from apps.api.app.workers.state_machine import InvalidScanTransition, ScanStateMachine
from packages.feature_engineering.schema import GameAggregateFeatures
from packages.llm import DeterministicExplainer, ExplanationEvidence, ExplanationInput
from packages.ml import (
    BaselineScorer,
    InsufficientGamesError,
    ModelMetadata,
    RiskThresholds,
    ScoringInput,
)


def make_game() -> GameAggregateFeatures:
    return GameAggregateFeatures(
        move_count=20,
        mean_centipawn_loss=20,
        median_centipawn_loss=18,
        p90_centipawn_loss=40,
        top_1_match_rate=0.8,
        top_3_match_rate=0.9,
        difficult_position_match_rate=0.8,
        tactical_error_count=1,
        centipawn_loss_stability=10,
        identical_time_share=0.4,
        median_time_seconds=2,
        mean_time_seconds=2,
        time_variance=0,
        time_difficulty_correlation=None,
        maia_divergence=0.2,
    )


def make_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_version="baseline-test",
        model_type="baseline",
        feature_schema_version="1.0.0",
        feature_names=["top_1_match_rate"],
        engine_name="stockfish",
        engine_version="fixture",
        analysis_depth=15,
        analysis_time_ms=100,
        thresholds=RiskThresholds(suspicious=0.6, high_risk=0.85),
    )


def test_scan_state_rejects_terminal_transition() -> None:
    with pytest.raises(InvalidScanTransition):
        ScanStateMachine.transition(DomainScanStatus.COMPLETED, DomainScanStatus.QUEUED)


def test_baseline_requires_ten_games() -> None:
    scorer = BaselineScorer(make_metadata())
    with pytest.raises(InsufficientGamesError):
        scorer.score(ScoringInput(username_normalized="tester", games=[make_game()] * 9))


def test_explanation_is_grounded_and_deterministic() -> None:
    data = ExplanationInput(
        risk_level="suspicious",
        risk_score=0.7,
        confidence_interval=(0.6, 0.8),
        games_analyzed=10,
        evidence=[
            ExplanationEvidence(
                code="TOP_MOVE_RATE",
                label="Top move rate",
                value=0.8,
                baseline=0.6,
                weight=0.3,
            )
        ],
        limitations=["Not proof of cheating"],
        model_version="baseline-test",
        prompt_version="v1",
    )
    output = DeterministicExplainer().explain(data)
    assert output.source == "deterministic"
    assert "TOP_MOVE_RATE" in output.summary
    assert output.limitations == data.limitations
