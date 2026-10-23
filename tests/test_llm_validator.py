"""Safety-gate tests: accusations, invented numbers, citations, reasoning."""

from __future__ import annotations

import pytest

from packages.llm import (
    ExplanationDirection,
    ExplanationEvidence,
    ExplanationInput,
    ExplanationOutput,
    ExplanationReasoning,
    ExplanationSource,
    ExplanationThresholds,
    ExplanationValidationError,
    load_knowledge,
    validate_explanation,
)

SUMMARY = (
    "Скор 0.62 при доверительном интервале 0.52–0.71 по 30 партиям "
    "соответствует повышенному риску. Наибольший вклад в сторону риска даёт "
    "top-1 (0.35), а CPL снижает его (0.21). Это статистический сигнал, который "
    "не является доказательством нарушения и требует дополнительной проверки."
)


def _data() -> ExplanationInput:
    return ExplanationInput(
        risk_level="suspicious",
        risk_score=0.62,
        confidence_interval=(0.52, 0.71),
        games_analyzed=30,
        evidence=[
            ExplanationEvidence(
                code="GAME_TOP_1_MATCH_RATE",
                label="Top-1 match rate",
                value=0.72,
                baseline=0.61,
                weight=0.4,
                contribution=0.35,
                contribution_share=0.62,
                direction=ExplanationDirection.SUPPORTS_RISK,
            ),
            ExplanationEvidence(
                code="GAME_MEAN_CENTIPAWN_LOSS",
                label="Mean centipawn loss",
                value=18.0,
                baseline=30.0,
                weight=0.3,
                contribution=-0.21,
                contribution_share=0.38,
                direction=ExplanationDirection.REDUCES_RISK,
            ),
        ],
        limitations=["Выборка небольшая, интервал широкий."],
        model_version="chessfraud-v1",
        prompt_version="rag-v1",
        thresholds=ExplanationThresholds(suspicious=0.4915, high_risk=0.5638),
    )


def _valid_output() -> ExplanationOutput:
    return ExplanationOutput(
        summary=SUMMARY,
        limitations=["Выборка небольшая, интервал широкий."],
        source="llm",
        reasoning=[
            ExplanationReasoning(
                evidence_code="GAME_TOP_1_MATCH_RATE",
                observation="Значение 0.72 при базовой линии 0.61, вклад 0.35.",
                interpretation="Признак поддерживает повышенный риск.",
                direction=ExplanationDirection.SUPPORTS_RISK,
                contribution=0.35,
            ),
            ExplanationReasoning(
                evidence_code="GAME_MEAN_CENTIPAWN_LOSS",
                observation="Значение 18 при базовой линии 30, вклад -0.21.",
                interpretation="Признак снижает риск.",
                direction=ExplanationDirection.REDUCES_RISK,
                contribution=-0.21,
            ),
        ],
        sources=[
            ExplanationSource(id="policy.no_accusation", title="Запрет обвинений"),
            ExplanationSource(id="policy.factual_only", title="Только факты"),
            ExplanationSource(
                id="thresholds.operating_points",
                title="Пороги вердикта",
            ),
        ],
    )


def _chunks():
    return load_knowledge().chunks


def test_accepts_a_valid_grounded_russian_output() -> None:
    validate_explanation(_valid_output(), _data(), _chunks())


@pytest.mark.parametrize("forbidden", ["читер", "винов", "доказывает", "cheating"])
def test_rejects_forbidden_accusation_words(forbidden: str) -> None:
    output = _valid_output().model_copy(
        update={"summary": f"Игрок не {forbidden}. {SUMMARY}"}
    )

    with pytest.raises(ExplanationValidationError, match="forbidden"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_negated_forbidden_word() -> None:
    output = _valid_output().model_copy(update={"summary": f"Не читер. {SUMMARY}"})

    with pytest.raises(ExplanationValidationError, match="forbidden"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_forbidden_word_in_limitations() -> None:
    output = _valid_output().model_copy(
        update={"limitations": ["Этот игрок — читер."]}
    )

    with pytest.raises(ExplanationValidationError, match="forbidden"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_unknown_number() -> None:
    output = _valid_output().model_copy(
        update={"summary": f"{SUMMARY} Точность модели 0.987654 на этой выборке."}
    )

    with pytest.raises(ExplanationValidationError, match="unverified number"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_missing_mandatory_source() -> None:
    output = _valid_output().model_copy(
        update={
            "sources": [
                ExplanationSource(id="policy.no_accusation", title="Запрет обвинений")
            ]
        }
    )

    with pytest.raises(ExplanationValidationError, match="mandatory sources"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_unretrieved_source() -> None:
    output = _valid_output().model_copy(
        update={
            "sources": [
                *_valid_output().sources,
                ExplanationSource(id="made.up", title="Invented"),
            ]
        }
    )

    with pytest.raises(ExplanationValidationError, match="not retrieved"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_unknown_evidence_code() -> None:
    reasoning = [
        _valid_output().reasoning[0].model_copy(
            update={"evidence_code": "GAME_UNKNOWN_FEATURE"}
        ),
        _valid_output().reasoning[1],
    ]
    output = _valid_output().model_copy(update={"reasoning": reasoning})

    with pytest.raises(ExplanationValidationError, match="unknown evidence code"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_missing_negation_phrase() -> None:
    summary = (
        "Скор 0.62 при доверительном интервале 0.52–0.71 по 30 партиям "
        "соответствует повышенному риску. Это статистический сигнал для "
        "дополнительной проверки."
    )
    output = _valid_output().model_copy(update={"summary": summary})

    with pytest.raises(ExplanationValidationError, match="non-accusation"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_bad_direction() -> None:
    reasoning = [
        _valid_output().reasoning[0].model_copy(update={"direction": "sideways"}),
        _valid_output().reasoning[1],
    ]
    output = _valid_output().model_copy(update={"reasoning": reasoning})

    with pytest.raises(ExplanationValidationError, match="invalid direction"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_mismatched_contribution() -> None:
    reasoning = [
        _valid_output().reasoning[0].model_copy(update={"contribution": 9.99}),
        _valid_output().reasoning[1],
    ]
    output = _valid_output().model_copy(update={"reasoning": reasoning})

    with pytest.raises(ExplanationValidationError, match="does not match"):
        validate_explanation(output, _data(), _chunks())


def test_rejects_empty_limitations_and_short_reasoning() -> None:
    output = _valid_output().model_copy(
        update={"limitations": [], "reasoning": [_valid_output().reasoning[0]]}
    )

    with pytest.raises(ExplanationValidationError, match="limitations"):
        validate_explanation(output, _data(), _chunks())
