"""Tests for the fourth pipeline stage that produces grounded explanations."""

import asyncio
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apps.api.app.config import Settings
from apps.api.app.db.base import Base
from apps.api.app.db.models import ScanModel, UserModel, VerdictModel
from apps.api.app.workers.analysis_stages import ExplanationStage
from apps.api.app.workers.llm_factory import create_explainer
from apps.api.app.workers.pipeline import ScanPipelineError
from packages.llm import (
    DeterministicExplainer,
    ExplanationDirection,
    ExplanationInput,
    ExplanationOutput,
    ExplanationReasoning,
    ExplanationSource,
    load_knowledge,
)

EVIDENCE = [
    {
        "code": "HIGH_TOP_MOVE_RATE",
        "label": "High top-1 move match rate",
        "value": 0.80,
        "baseline": 0.61,
        "weight": 0.35,
        "contribution": 0.0665,
        "contribution_share": 0.7,
        "direction": "supports_risk",
    },
    {
        "code": "LOW_CENTIPAWN_LOSS",
        "label": "Low mean centipawn loss",
        "value": 20.0,
        "baseline": 35.0,
        "weight": 0.30,
        "contribution": -0.03,
        "contribution_share": 0.3,
        "direction": "reduces_risk",
    },
]
THRESHOLDS = {"suspicious": 0.55, "high_risk": 0.75}


async def _new_database() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_verdict(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    games_analyzed: int = 12,
    with_verdict: bool = True,
) -> str:
    """Create a user, an explaining scan, and optionally its verdict."""

    user = UserModel(
        id=str(uuid4()),
        lichess_username="Tester",
        username_normalized="tester",
    )
    scan = ScanModel(
        id=str(uuid4()),
        user_id=user.id,
        status="explaining",
        requested_time_control="blitz",
        requested_game_limit=max(10, games_analyzed),
        rated_only=True,
        games_analyzed=games_analyzed,
        created_at=datetime.utcnow(),
    )
    async with session_factory() as session:
        session.add_all([user, scan])
        await session.flush()
        if with_verdict:
            session.add(
                VerdictModel(
                    id=str(uuid4()),
                    scan_id=scan.id,
                    risk_level="suspicious",
                    risk_score=0.68,
                    confidence_lower=0.60,
                    confidence_upper=0.76,
                    summary="Инициальное детерминированное объяснение.",
                    evidence=[dict(item) for item in EVIDENCE],
                    limitations=["Automated analysis is not proof of cheating."],
                    thresholds=dict(THRESHOLDS),
                    explanation=None,
                    model_version="baseline-fixture",
                    prompt_version="deterministic-v1",
                )
            )
        await session.commit()
    return scan.id


async def _verdict(
    session_factory: async_sessionmaker[AsyncSession],
    scan_id: str,
) -> VerdictModel | None:
    async with session_factory() as session:
        return await session.scalar(
            select(VerdictModel).where(VerdictModel.scan_id == scan_id)
        )


class StubLLMExplainer:
    """Return a valid provider-shaped explanation without any network call."""

    def __init__(self) -> None:
        self.calls = 0

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        self.calls += 1
        first = data.evidence[0]
        mandatory = load_knowledge().mandatory
        return ExplanationOutput(
            summary=(
                "Итог: риск-скор указывает на повышенный риск, но вывод "
                "не является доказательством нарушения."
            ),
            limitations=["Автоматический анализ не является доказательством нарушения."],
            source="llm",
            reasoning=[
                ExplanationReasoning(
                    evidence_code=first.code,
                    observation=f"{first.label}: значение {first.value:.2f}.",
                    interpretation="Признак сдвигает решение модели в сторону риска.",
                    direction=ExplanationDirection.SUPPORTS_RISK,
                    contribution=first.contribution,
                )
            ],
            sources=[
                ExplanationSource(id=chunk.id, title=chunk.title)
                for chunk in mandatory[:2]
            ],
            provider="openai",
            model="gpt-4o-mini",
            prompt_version="rag-v1",
            knowledge_version="1.0.0",
        )


class ExplodingExplainer:
    """Simulate an unexpected bug rather than a handled provider failure."""

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        raise RuntimeError("explainer crashed")


class SequenceExplainer:
    """Return a different summary on every run to prove updates are in place."""

    def __init__(self) -> None:
        self.calls = 0

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        self.calls += 1
        return ExplanationOutput(
            summary=(
                f"Итог: вариант {self.calls}. Результат не является доказательством."
            ),
            limitations=["Автоматический анализ не является доказательством нарушения."],
            source="llm",
            prompt_version="rag-v1",
        )


def test_explanation_stage_persists_llm_output() -> None:
    async def scenario() -> None:
        engine, session_factory = await _new_database()
        scan_id = await _seed_verdict(session_factory)
        explainer = StubLLMExplainer()

        await ExplanationStage(session_factory, explainer).run(UUID(scan_id))
        assert explainer.calls == 1

        verdict = await _verdict(session_factory, scan_id)
        assert verdict is not None
        assert verdict.summary.startswith("Итог:")
        assert verdict.prompt_version == "rag-v1"
        assert verdict.explanation is not None
        assert verdict.explanation["source"] == "llm"
        assert verdict.explanation["provider"] == "openai"
        assert verdict.explanation["model"] == "gpt-4o-mini"
        assert verdict.explanation["knowledge_version"] == "1.0.0"
        assert verdict.explanation["reasoning"][0]["evidence_code"] == (
            "HIGH_TOP_MOVE_RATE"
        )
        assert verdict.explanation["sources"]
        # The numeric verdict is owned by scoring and must never change here.
        assert verdict.risk_score == pytest.approx(0.68)
        assert verdict.evidence == EVIDENCE
        assert verdict.model_version == "baseline-fixture"
        await engine.dispose()

    asyncio.run(scenario())


def test_explainer_crash_falls_back_without_failing_the_stage() -> None:
    async def scenario() -> None:
        engine, session_factory = await _new_database()
        scan_id = await _seed_verdict(session_factory)

        await ExplanationStage(session_factory, ExplodingExplainer()).run(UUID(scan_id))

        verdict = await _verdict(session_factory, scan_id)
        assert verdict is not None
        assert verdict.summary.startswith("Итог:")
        assert verdict.explanation is not None
        assert verdict.explanation["source"] == "deterministic"
        assert verdict.explanation["fallback_reason"] == "unexpected_error"
        assert verdict.prompt_version == "deterministic-v1"
        await engine.dispose()

    asyncio.run(scenario())


def test_missing_verdict_raises_explicit_pipeline_error() -> None:
    async def scenario() -> None:
        engine, session_factory = await _new_database()
        scan_id = await _seed_verdict(session_factory, with_verdict=False)

        with pytest.raises(ScanPipelineError) as raised:
            await ExplanationStage(session_factory, DeterministicExplainer()).run(
                UUID(scan_id)
            )
        assert raised.value.code == "VERDICT_NOT_FOUND"
        await engine.dispose()

    asyncio.run(scenario())


def test_rerun_updates_the_existing_verdict_without_duplicates() -> None:
    async def scenario() -> None:
        engine, session_factory = await _new_database()
        scan_id = await _seed_verdict(session_factory)
        explainer = SequenceExplainer()
        stage = ExplanationStage(session_factory, explainer)

        await stage.run(UUID(scan_id))
        first = await _verdict(session_factory, scan_id)
        assert first is not None
        first_id = first.id

        await stage.run(UUID(scan_id))
        second = await _verdict(session_factory, scan_id)
        assert second is not None
        assert explainer.calls == 2
        assert second.id == first_id
        assert "вариант 2" in second.summary

        async with session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(VerdictModel)
            )
        assert count == 1
        await engine.dispose()

    asyncio.run(scenario())


def test_create_explainer_without_key_is_deterministic() -> None:
    settings = Settings(llm_enabled=True, llm_api_key="", llm_provider="openai")
    assert isinstance(create_explainer(settings), DeterministicExplainer)


def test_create_explainer_with_invalid_provider_falls_back() -> None:
    settings = Settings(llm_provider="unknown-provider", llm_api_key="secret")
    assert isinstance(create_explainer(settings), DeterministicExplainer)
