"""Deterministic keyword/tag retrieval over the grounded JSON knowledge base.

There are no embeddings, no vector database, and no network calls here: the
knowledge base is a small, versioned JSON file, and retrieval is a pure function
of its contents plus the verdict payload.  That keeps explanations reproducible
and lets tests assert an exact chunk order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .contracts import ExplanationInput

DEFAULT_KNOWLEDGE_PATH = Path(__file__).with_name("knowledge.json")

#: Chunks that must exist in every knowledge file.
REQUIRED_CHUNK_IDS: frozenset[str] = frozenset(
    {
        "policy.no_accusation",
        "policy.factual_only",
        "policy.uncertainty",
        "policy.privacy",
        "policy.style",
        "methodology.score",
        "features.definitions",
        "limitations.cohort",
        "thresholds.operating_points",
        "faq.misinterpretations",
    }
)

#: Chunks injected into every prompt regardless of the query, in this order.
MANDATORY_CHUNK_IDS: tuple[str, ...] = (
    "policy.no_accusation",
    "policy.factual_only",
    "policy.uncertainty",
    "policy.privacy",
    "limitations.cohort",
)

#: Methodology vocabulary always added to the retrieval query.
_METHODOLOGY_TERMS: tuple[str, ...] = (
    "confidence interval",
    "sample size",
    "prevalence",
    "threshold",
    "methodology",
    "risk score",
    "contribution",
)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "и",
        "в",
        "во",
        "для",
        "из",
        "к",
        "как",
        "на",
        "не",
        "о",
        "об",
        "от",
        "по",
        "с",
        "у",
        "что",
        "это",
    }
)


class KnowledgeChunk(BaseModel):
    """One grounded, citable fragment of the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    always: bool = False
    priority: float = 0.0


class KnowledgeBase(BaseModel):
    """Validated knowledge file with unique IDs and all required chunks."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    chunks: list[KnowledgeChunk] = Field(min_length=1)
    description: str | None = None

    @model_validator(mode="after")
    def unique_and_required_chunks(self) -> KnowledgeBase:
        ids = [chunk.id for chunk in self.chunks]
        duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate knowledge chunk ids: {duplicates}")
        missing = sorted(REQUIRED_CHUNK_IDS - set(ids))
        if missing:
            raise ValueError(f"knowledge base is missing required chunks: {missing}")
        return self

    def by_id(self, chunk_id: str) -> KnowledgeChunk | None:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    @property
    def mandatory(self) -> list[KnowledgeChunk]:
        """Mandatory chunks in their canonical order (missing IDs are fatal)."""

        chunks: list[KnowledgeChunk] = []
        for chunk_id in MANDATORY_CHUNK_IDS:
            chunk = self.by_id(chunk_id)
            if chunk is None:
                raise ValueError(f"knowledge base is missing mandatory chunk {chunk_id!r}")
            chunks.append(chunk)
        return chunks


def load_knowledge(path: str | Path | None = None) -> KnowledgeBase:
    """Load and validate ``knowledge.json`` from ``path`` or the package default."""

    source = Path(path) if path is not None else DEFAULT_KNOWLEDGE_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return KnowledgeBase.model_validate(payload)


def build_query(data: ExplanationInput) -> str:
    """Build a deterministic retrieval query from grounded verdict fields.

    Identity is not present in ``ExplanationInput`` by contract, so it cannot
    leak into the query.
    """

    parts: list[str] = [data.risk_level, *_METHODOLOGY_TERMS]
    for item in data.evidence:
        parts.extend((item.code, item.label))
        if item.direction is not None:
            parts.append(str(item.direction))
    return " ".join(parts)


def retrieve(
    knowledge: KnowledgeBase,
    query: str,
    *,
    max_chunks: int = 6,
) -> list[KnowledgeChunk]:
    """Return mandatory chunks plus the best keyword/tag matches.

    Ranking is deterministic: tag matches weigh more than body-token overlap,
    then the chunk priority is added as a small tie-breaker, and ties sort by ID.
    The result never exceeds ``max(max_chunks, len(mandatory))``; when the budget
    is tight the mandatory policy/limitations chunks always win.
    """

    mandatory = knowledge.mandatory
    budget = max(max_chunks, len(mandatory))
    slots = budget - len(mandatory)
    if slots <= 0:
        return mandatory

    query_tokens = _tokens(query)
    scored: list[tuple[float, str, KnowledgeChunk]] = []
    mandatory_ids = {chunk.id for chunk in mandatory}
    for chunk in knowledge.chunks:
        if chunk.id in mandatory_ids:
            continue
        body_tokens = _tokens(f"{chunk.title} {chunk.text}")
        tag_tokens = _tokens(" ".join(chunk.tags))
        overlap = len(query_tokens & body_tokens)
        tag_hits = len(query_tokens & tag_tokens)
        score = 2.0 * tag_hits + 1.0 * overlap + 0.05 * chunk.priority
        if score <= 0.0:
            continue
        scored.append((score, chunk.id, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [*mandatory, *(chunk for _score, _id, chunk in scored[:slots])]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }


def chunk_ids(chunks: Iterable[KnowledgeChunk]) -> set[str]:
    """Convenience helper used by the validator and tests."""

    return {chunk.id for chunk in chunks}


__all__ = [
    "DEFAULT_KNOWLEDGE_PATH",
    "MANDATORY_CHUNK_IDS",
    "REQUIRED_CHUNK_IDS",
    "KnowledgeBase",
    "KnowledgeChunk",
    "build_query",
    "chunk_ids",
    "load_knowledge",
    "retrieve",
]
