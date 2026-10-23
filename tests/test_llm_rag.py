"""Deterministic JSON knowledge retrieval tests."""

from __future__ import annotations

from packages.llm import (
    ExplanationEvidence,
    ExplanationInput,
    KnowledgeBase,
    build_query,
    load_knowledge,
    retrieve,
)
from packages.llm.rag import MANDATORY_CHUNK_IDS, REQUIRED_CHUNK_IDS


def _data() -> ExplanationInput:
    return ExplanationInput(
        risk_level="suspicious",
        risk_score=0.56,
        confidence_interval=(0.47, 0.65),
        games_analyzed=30,
        evidence=[
            ExplanationEvidence(
                code="GAME_TOP_1_MATCH_RATE",
                label="Top-1 match rate",
                value=0.7,
                baseline=0.61,
                weight=0.4,
                contribution=0.12,
                contribution_share=0.6,
            )
        ],
        limitations=["Выборка небольшая."],
        model_version="chessfraud-v1",
        prompt_version="rag-v1",
    )


def test_knowledge_file_loads_and_is_complete() -> None:
    knowledge = load_knowledge()

    assert knowledge.version == "1.0.0"
    assert len(knowledge.chunks) == len({chunk.id for chunk in knowledge.chunks})
    assert REQUIRED_CHUNK_IDS <= {chunk.id for chunk in knowledge.chunks}
    assert [chunk.id for chunk in knowledge.mandatory] == list(MANDATORY_CHUNK_IDS)
    assert all(chunk.text.strip() for chunk in knowledge.chunks)


def test_mandatory_chunks_are_always_retrieved() -> None:
    knowledge = load_knowledge()

    for query in ("", "low_risk", "совершенно другой запрос"):
        chunks = retrieve(knowledge, query, max_chunks=4)
        ids = {chunk.id for chunk in chunks}
        assert set(MANDATORY_CHUNK_IDS) <= ids


def test_top_k_budget_and_deterministic_order() -> None:
    knowledge = load_knowledge()
    query = build_query(_data())

    first = retrieve(knowledge, query, max_chunks=6)
    second = retrieve(knowledge, query, max_chunks=6)
    larger = retrieve(knowledge, query, max_chunks=8)

    assert len(first) <= 6
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len(larger) >= len(first)
    assert len(larger) <= 8


def test_relevant_chunk_beats_unrelated_chunk() -> None:
    knowledge = load_knowledge()

    chunks = retrieve(knowledge, "threshold suspicious high_risk risk_level", max_chunks=6)

    optional = [chunk.id for chunk in chunks if chunk.id not in MANDATORY_CHUNK_IDS]
    assert optional == ["thresholds.operating_points"]


def test_build_query_uses_verdict_fields_only() -> None:
    query = build_query(_data())

    assert "suspicious" in query
    assert "GAME_TOP_1_MATCH_RATE" in query
    assert "confidence interval" in query
    assert "prevalence" in query
    assert "somebody" not in query


def test_duplicate_or_missing_chunks_are_rejected() -> None:
    knowledge = load_knowledge()
    payload = knowledge.model_dump()

    duplicate = dict(payload)
    duplicate["chunks"] = [*payload["chunks"], payload["chunks"][0]]
    try:
        KnowledgeBase.model_validate(duplicate)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover - explicit failure is the point
        raise AssertionError("duplicate chunk ids must be rejected")

    missing = dict(payload)
    missing["chunks"] = [
        chunk for chunk in payload["chunks"] if chunk["id"] != "policy.privacy"
    ]
    try:
        KnowledgeBase.model_validate(missing)
    except ValueError as exc:
        assert "missing required chunks" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing mandatory chunks must be rejected")
