"""Contract defaults and privacy invariants for the LLM layer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from packages.llm import (
    ExplanationDirection,
    ExplanationEvidence,
    ExplanationInput,
    ExplanationOutput,
    ExplanationReasoning,
    ExplanationThresholds,
)

_IDENTITY_FIELDS = {
    "username",
    "username_normalized",
    "pgn",
    "game_id",
    "game_ids",
    "player_id",
    "player_name",
    "opponent",
    "rating",
    "elo",
}


def _evidence() -> ExplanationEvidence:
    return ExplanationEvidence(
        code="GAME_TOP_1_MATCH_RATE",
        label="Top-1 match rate",
        value=0.7,
        baseline=0.61,
        weight=0.4,
    )


def _input() -> ExplanationInput:
    return ExplanationInput(
        risk_level="suspicious",
        risk_score=0.56,
        confidence_interval=(0.47, 0.65),
        games_analyzed=30,
        evidence=[_evidence()],
        limitations=["Выборка небольшая."],
        model_version="chessfraud-v1",
        prompt_version="rag-v1",
    )


def test_explanation_output_defaults_keep_existing_callers_working() -> None:
    output = ExplanationOutput(
        summary="Это статистический сигнал, а не приговор.",
        limitations=["Не является доказательством нарушения."],
        source="deterministic",
    )

    assert output.reasoning == []
    assert output.sources == []
    assert output.provider is None
    assert output.model is None
    assert output.prompt_version == "deterministic-v1"
    assert output.knowledge_version == "1.0.0"
    assert output.fallback_reason is None


def test_direction_enum_values() -> None:
    assert {item.value for item in ExplanationDirection} == {
        "supports_risk",
        "reduces_risk",
        "neutral",
    }
    assert ExplanationDirection("supports_risk") is ExplanationDirection.SUPPORTS_RISK


def test_evidence_contribution_fields_are_optional() -> None:
    evidence = _evidence()

    assert evidence.contribution is None
    assert evidence.contribution_share is None
    assert evidence.direction is None

    enriched = evidence.model_copy(
        update={
            "contribution": 0.12,
            "contribution_share": 0.4,
            "direction": ExplanationDirection.SUPPORTS_RISK,
        }
    )
    assert enriched.direction is ExplanationDirection.SUPPORTS_RISK


def test_explanation_input_has_no_identity_fields() -> None:
    model_fields = set(ExplanationInput.model_fields)

    assert _IDENTITY_FIELDS.isdisjoint(model_fields)
    assert {"risk_level", "risk_score", "evidence", "model_version"} <= model_fields


def test_explanation_input_rejects_identity_payload() -> None:
    payload = _input().model_dump()
    payload["username"] = "somebody"

    with pytest.raises(ValidationError):
        ExplanationInput.model_validate(payload)


def test_reasoning_and_thresholds_contracts() -> None:
    reasoning = ExplanationReasoning(
        evidence_code="GAME_TOP_1_MATCH_RATE",
        observation="Значение 0.7 выше базовой линии.",
        interpretation="Признак поддерживает риск.",
        direction=ExplanationDirection.SUPPORTS_RISK,
        contribution=0.12,
    )
    thresholds = ExplanationThresholds(suspicious=0.4915, high_risk=0.5638)

    assert reasoning.contribution == 0.12
    assert thresholds.high_risk > thresholds.suspicious

    with pytest.raises(ValidationError):
        ExplanationThresholds(suspicious=0.6, high_risk=0.5)
