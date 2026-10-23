"""Safety gate for LLM explanations.

The validator is intentionally strict and independent of the provider: it checks
that the model did not accuse anyone, did not invent numbers, stayed on the
provided evidence codes, matched the numeric contributions, cited retrieved
knowledge, and kept the verdict non-accusatory.  Any failure raises
:class:`ExplanationValidationError`, which the explainer converts into a
deterministic fallback.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .contracts import (
    ExplanationDirection,
    ExplanationInput,
    ExplanationOutput,
)
from .rag import KnowledgeChunk

#: Accusation stems that must never appear in generated prose.  The noun
#: "доказательство" is not here: the required negation phrase needs it.
FORBIDDEN_STEMS: tuple[str, ...] = (
    "читер",
    "винов",
    "доказано",
    "доказывает",
    "обман",
    "жулик",
    "бан",
    "наказание",
    "санкц",
    "cheater",
    "cheating",
    "guilty",
    "proven",
    "ban",
    "sanction",
)

#: At least one of these must be present in the summary.
REQUIRED_NEGATIONS: tuple[str, ...] = (
    "не доказательств",
    "не является доказательством",
    "не доказательство",
)

#: Numbers that are harmless formatting/sentence-count values.
FORMATTING_WHITELIST: frozenset[float] = frozenset(
    {0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 95.0, 100.0}
)

_NUMBER_RE = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)")

_EVIDENCE_SOURCE_IDS = frozenset({"policy.no_accusation", "policy.factual_only"})


class ExplanationValidationError(ValueError):
    """Raised when an explanation violates the safety or grounding contract."""


def validate_explanation(
    output: ExplanationOutput,
    data: ExplanationInput,
    chunks: list[KnowledgeChunk],
) -> None:
    """Raise :class:`ExplanationValidationError` on the first violation."""

    _validate_shape(output, data)
    _validate_forbidden_words(output)
    _validate_negation(output)
    _validate_numbers(output, data, chunks)
    _validate_reasoning(output, data)
    _validate_sources(output, chunks)


def _validate_shape(output: ExplanationOutput, data: ExplanationInput) -> None:
    summary = output.summary.strip()
    if not 40 <= len(summary) <= 2000:
        raise ExplanationValidationError(
            "summary length must be between 40 and 2000 characters"
        )
    if not output.limitations or not any(item.strip() for item in output.limitations):
        raise ExplanationValidationError("limitations must be a non-empty list")
    if not 2 <= len(output.reasoning) <= 6:
        raise ExplanationValidationError("reasoning must contain 2 to 6 items")
    # Keep the score owned by the numeric model.  This is cheap insurance if an
    # output object was built by hand rather than parsed from provider JSON.
    if not 0.0 <= data.risk_score <= 1.0:
        raise ExplanationValidationError("input risk_score is outside [0, 1]")


def _validate_forbidden_words(output: ExplanationOutput) -> None:
    texts = [output.summary, *output.limitations]
    for item in output.reasoning:
        texts.extend((item.observation, item.interpretation))
    combined = " ".join(texts).casefold()
    for stem in FORBIDDEN_STEMS:
        if stem in combined:
            raise ExplanationValidationError(
                f"forbidden accusation stem {stem!r} appears in the explanation"
            )


def _validate_negation(output: ExplanationOutput) -> None:
    summary = output.summary.casefold()
    if not any(phrase in summary for phrase in REQUIRED_NEGATIONS):
        raise ExplanationValidationError(
            "summary must contain a non-accusation phrase such as "
            "'не является доказательством'"
        )


def _validate_numbers(
    output: ExplanationOutput,
    data: ExplanationInput,
    chunks: list[KnowledgeChunk],
) -> None:
    allowed = _allowed_numbers(data, chunks)
    texts = [output.summary]
    for item in output.reasoning:
        texts.extend((item.observation, item.interpretation))
    for text in texts:
        for raw in _NUMBER_RE.findall(text):
            number = float(raw.replace(",", "."))
            if not _number_allowed(number, allowed, raw=raw):
                raise ExplanationValidationError(
                    f"unverified number {raw!r} is not present in the verdict or sources"
                )


def _validate_reasoning(output: ExplanationOutput, data: ExplanationInput) -> None:
    evidence_by_code = {item.code: item for item in data.evidence}
    for item in output.reasoning:
        if not item.observation.strip() or not item.interpretation.strip():
            raise ExplanationValidationError(
                "reasoning observation and interpretation must not be empty"
            )
        if item.evidence_code not in evidence_by_code:
            raise ExplanationValidationError(
                f"reasoning refers to unknown evidence code {item.evidence_code!r}"
            )
        if not isinstance(item.direction, ExplanationDirection):
            raise ExplanationValidationError(
                f"invalid direction for evidence code {item.evidence_code!r}"
            )
        expected = evidence_by_code[item.evidence_code].contribution
        if item.contribution is not None:
            if expected is None:
                raise ExplanationValidationError(
                    f"reasoning claims a contribution for {item.evidence_code!r} "
                    "but the verdict has none"
                )
            if not math.isclose(
                item.contribution,
                expected,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ExplanationValidationError(
                    f"contribution for {item.evidence_code!r} does not match the verdict"
                )


def _validate_sources(
    output: ExplanationOutput,
    chunks: list[KnowledgeChunk],
) -> None:
    if not output.sources:
        raise ExplanationValidationError("sources must not be empty")
    available = {chunk.id for chunk in chunks}
    cited = {source.id for source in output.sources}
    unknown = sorted(cited - available)
    if unknown:
        raise ExplanationValidationError(
            f"explanation cites sources that were not retrieved: {unknown}"
        )
    missing = sorted(_EVIDENCE_SOURCE_IDS - cited)
    if missing:
        raise ExplanationValidationError(
            f"explanation must cite mandatory sources: {missing}"
        )


def _allowed_numbers(
    data: ExplanationInput,
    chunks: Iterable[KnowledgeChunk],
) -> list[float]:
    values: list[float] = [
        data.risk_score,
        *data.confidence_interval,
        float(data.games_analyzed),
        *FORMATTING_WHITELIST,
    ]
    if data.thresholds is not None:
        values.extend((data.thresholds.suspicious, data.thresholds.high_risk))
    for item in data.evidence:
        values.extend(
            (
                item.value,
                item.baseline,
                item.weight,
            )
        )
        for extra in (item.contribution, item.contribution_share):
            if extra is not None:
                values.append(extra)
                values.append(abs(extra))
    for chunk in chunks:
        for raw in _NUMBER_RE.findall(f"{chunk.title} {chunk.text}"):
            values.append(float(raw.replace(",", ".")))
    return values


def _number_allowed(
    candidate: float,
    allowed: Iterable[float],
    *,
    raw: str,
) -> bool:
    decimals = len(raw.replace(",", ".").partition(".")[2])
    for value in allowed:
        if _close(candidate, value):
            return True
        # Percent rendering of a probability or share, e.g. 40.3% vs 0.403.
        if _close(candidate / 100.0, value):
            return True
        # Rounded narrative, e.g. 0.73 for a threshold of 0.7303.  Only round to
        # the precision the model actually printed, so a long invented decimal
        # cannot hide behind a coarse comparison.
        if 0 < decimals <= 3 and round(candidate, decimals) == round(value, decimals):
            return True
    return False


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-3, abs_tol=5e-3)


__all__ = [
    "ExplanationValidationError",
    "FORBIDDEN_STEMS",
    "FORMATTING_WHITELIST",
    "REQUIRED_NEGATIONS",
    "validate_explanation",
]
