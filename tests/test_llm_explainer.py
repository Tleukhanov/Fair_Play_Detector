"""Explainer orchestration: retrieval, retry, validation, and fallback."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from packages.llm import (
    DeterministicExplainer,
    ExplanationDirection,
    ExplanationEvidence,
    ExplanationInput,
    ExplanationThresholds,
    LLMConfig,
    LLMExplainer,
    ProviderTimeout,
    build_explainer,
    load_knowledge,
)
from packages.llm.rag import DEFAULT_KNOWLEDGE_PATH

SUMMARY = (
    "Скор 0.62 при доверительном интервале 0.52–0.71 по 30 партиям "
    "соответствует повышенному риску. Наибольший вклад в сторону риска даёт "
    "top-1 (0.35), а CPL снижает его (0.21). Это статистический сигнал, который "
    "не является доказательством нарушения и требует дополнительной проверки."
)

VALID_PAYLOAD: dict[str, Any] = {
    "summary": SUMMARY,
    "reasoning": [
        {
            "evidence_code": "GAME_TOP_1_MATCH_RATE",
            "observation": "Значение 0.72 при базовой линии 0.61, вклад 0.35.",
            "interpretation": "Признак поддерживает повышенный риск.",
            "direction": "supports_risk",
            "contribution": 0.35,
        },
        {
            "evidence_code": "GAME_MEAN_CENTIPAWN_LOSS",
            "observation": "Значение 18 при базовой линии 30, вклад -0.21.",
            "interpretation": "Признак снижает риск.",
            "direction": "reduces_risk",
            "contribution": -0.21,
        },
    ],
    "sources": [
        {"id": "policy.no_accusation", "title": "Запрет обвинений"},
        {"id": "policy.factual_only", "title": "Только проверяемые факты"},
    ],
    "limitations": ["Выборка небольшая, интервал широкий."],
}


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


class FakeProvider:
    """Deterministic async provider double; records every call."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("fake provider ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response


def _config() -> LLMConfig:
    return LLMConfig(
        enabled=True,
        provider="openai",
        api_key="test-key",
        model="test-model",
        knowledge_path=DEFAULT_KNOWLEDGE_PATH,
    )


def _explainer(provider: FakeProvider) -> LLMExplainer:
    return LLMExplainer(
        _config(),
        provider=provider,
        knowledge=load_knowledge(),
    )


def test_valid_provider_json_becomes_llm_output() -> None:
    provider = FakeProvider([json.dumps(VALID_PAYLOAD, ensure_ascii=False)])
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "llm"
    assert output.provider == "openai"
    assert output.model == "test-model"
    assert output.prompt_version == "rag-v1"
    assert output.knowledge_version == "1.0.0"
    assert output.fallback_reason is None
    assert output.summary == SUMMARY
    assert len(output.reasoning) == 2
    assert {source.id for source in output.sources} >= {
        "policy.no_accusation",
        "policy.factual_only",
    }
    assert len(provider.calls) == 1
    user_message = provider.calls[0][1]["content"]
    assert "GAME_TOP_1_MATCH_RATE" in user_message
    assert "knowledge_chunks" in user_message


def test_invalid_json_retries_then_falls_back() -> None:
    provider = FakeProvider(["not json", "still not json"])
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "deterministic"
    assert output.fallback_reason == "invalid_json"
    assert "не является доказательством" in output.summary
    assert len(provider.calls) == 2


def test_forbidden_phrase_retries_then_falls_back() -> None:
    forbidden = dict(VALID_PAYLOAD)
    forbidden["summary"] = "Нарушитель — читер. " + SUMMARY
    provider = FakeProvider(
        [json.dumps(forbidden, ensure_ascii=False), json.dumps(forbidden, ensure_ascii=False)]
    )
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "deterministic"
    assert output.fallback_reason == "validation_failed"
    assert len(provider.calls) == 2


def test_repair_retry_can_recover() -> None:
    provider = FakeProvider(
        ["not json", json.dumps(VALID_PAYLOAD, ensure_ascii=False)]
    )
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "llm"
    assert output.fallback_reason is None
    assert len(provider.calls) == 2
    assert provider.calls[1][-1]["role"] == "user"
    assert "отклонён" in provider.calls[1][-1]["content"]


def test_provider_error_falls_back_without_raising() -> None:
    provider = FakeProvider([RuntimeError("boom")])
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "deterministic"
    assert output.fallback_reason == "provider_error"
    assert len(provider.calls) == 1


def test_provider_timeout_falls_back_with_timeout_reason() -> None:
    provider = FakeProvider([ProviderTimeout("slow")])
    output = asyncio.run(_explainer(provider).aexplain(_data()))

    assert output.source == "deterministic"
    assert output.fallback_reason == "timeout"


def test_build_explainer_without_key_is_deterministic() -> None:
    for config in (
        LLMConfig(api_key=""),
        LLMConfig(enabled=False, api_key="secret"),
    ):
        explainer = build_explainer(config)
        assert isinstance(explainer, DeterministicExplainer)


def test_llm_explainer_without_key_reports_no_api_key() -> None:
    provider = FakeProvider([])
    explainer = LLMExplainer(
        LLMConfig(api_key="", provider="openai", model="test-model"),
        provider=provider,
        knowledge=load_knowledge(),
    )
    output = asyncio.run(explainer.aexplain(_data()))

    assert output.source == "deterministic"
    assert output.fallback_reason == "no_api_key"
    assert provider.calls == []


def test_build_explainer_with_key_is_llm() -> None:
    explainer = build_explainer(_config(), knowledge=load_knowledge())

    assert isinstance(explainer, LLMExplainer)


def test_deterministic_fallback_is_sync_and_async() -> None:
    explainer = DeterministicExplainer()
    sync_output = explainer.explain(_data())
    async_output = asyncio.run(explainer.aexplain(_data()))

    assert sync_output.source == "deterministic"
    assert async_output.model_dump() == sync_output.model_dump()
    assert "не является доказательством" in sync_output.summary
    assert sync_output.prompt_version == "deterministic-v1"
    assert sync_output.knowledge_version == "1.0.0"
