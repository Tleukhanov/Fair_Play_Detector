"""Safe deterministic explanation used when no LLM provider is configured.

The fallback is not a generic disclaimer: it reads the same grounded evidence as
the LLM path and presents the strongest risk-increasing and risk-reducing
contributions (or evidence values against their baselines), the confidence
interval, and the sample size.  It never invents a number and never changes the
score.  It is available both synchronously (backward compatibility) and
asynchronously (the explainer protocol).
"""

from __future__ import annotations

from .contracts import (
    ExplanationDirection,
    ExplanationEvidence,
    ExplanationInput,
    ExplanationOutput,
    ExplanationReasoning,
)

#: Version stored with every deterministic explanation.
DETERMINISTIC_PROMPT_VERSION = "deterministic-v1"
KNOWLEDGE_VERSION = "1.0.0"

_RISK_LABELS: dict[str, str] = {
    "low_risk": "низкий риск",
    "suspicious": "повышенный риск",
    "high_risk": "высокий риск",
}

_NO_PROOF_MARKERS: tuple[str, ...] = (
    "не является доказательств",
    "не доказательств",
    "not proof",
    "not evidence",
    "не является доказательством",
)


class DeterministicExplainer:
    """Generate a grounded Russian explanation without any external call."""

    def explain(self, data: ExplanationInput) -> ExplanationOutput:
        """Return a reproducible explanation for ``data``."""

        ranked = sorted(
            data.evidence,
            key=lambda item: (
                -abs(item.contribution) if item.contribution is not None else 0.0,
                -abs(item.weight),
                item.code,
            ),
        )
        summary = self._summary(data, ranked)
        return ExplanationOutput(
            summary=summary,
            limitations=self._limitations(data),
            source="deterministic",
            reasoning=self._reasoning(data, ranked),
            sources=[],
            provider=None,
            model=data.model_version,
            prompt_version=DETERMINISTIC_PROMPT_VERSION,
            knowledge_version=KNOWLEDGE_VERSION,
        )

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        """Async wrapper so the deterministic and LLM explainers share a protocol."""

        return self.explain(data)

    def _summary(
        self,
        data: ExplanationInput,
        ranked: list[ExplanationEvidence],
    ) -> str:
        label = _RISK_LABELS.get(data.risk_level, data.risk_level)
        lower, upper = data.confidence_interval
        parts = [
            f"Итог: {label}; риск-скор {data.risk_score:.2f} по "
            f"{data.games_analyzed} проанализированным партиям."
        ]

        codes = ", ".join(item.code for item in data.evidence) or "нет"
        parts.append(f"Основание: признаки {codes}.")

        increasing = [
            item
            for item in ranked
            if _direction_for(item) is ExplanationDirection.SUPPORTS_RISK
        ]
        reducing = [
            item
            for item in ranked
            if _direction_for(item) is ExplanationDirection.REDUCES_RISK
        ]
        if increasing or reducing:
            fragments: list[str] = []
            if increasing:
                top = increasing[0]
                fragments.append(
                    f"в сторону риска сильнее всего влияет {top.label} "
                    f"({top.code}, вклад {_contribution_text(top)})"
                )
            if reducing:
                top = reducing[0]
                fragments.append(
                    f"в сторону снижения риска — {top.label} "
                    f"({top.code}, вклад {_contribution_text(top)})"
                )
            parts.append("По вкладу признаков " + "; ".join(fragments) + ".")
        elif ranked:
            item = ranked[0]
            parts.append(
                f"Самый весомый признак — {item.label} ({item.code}): "
                f"значение {item.value:.2f} при базовой линии {item.baseline:.2f}."
            )

        parts.append(
            f"Доверительный интервал {lower:.2f}–{upper:.2f}; "
            "при широком интервале или малом числе партий вывод неустойчив."
        )
        parts.append(
            "Это статистический сигнал для дополнительной проверки: "
            "результат не является доказательством нарушения."
        )
        return " ".join(parts)

    def _reasoning(
        self,
        data: ExplanationInput,
        ranked: list[ExplanationEvidence],
    ) -> list[ExplanationReasoning]:
        items: list[ExplanationReasoning] = []
        for evidence in ranked[:4]:
            direction = _direction_for(evidence)
            items.append(
                ExplanationReasoning(
                    evidence_code=evidence.code,
                    observation=(
                        f"{evidence.label}: значение {evidence.value:.2f}, "
                        f"базовая линия {evidence.baseline:.2f}."
                    ),
                    interpretation=_interpretation(evidence, direction),
                    direction=direction,
                    contribution=evidence.contribution,
                )
            )
        if len(items) == 1:
            first = items[0]
            items.append(
                ExplanationReasoning(
                    evidence_code=first.evidence_code,
                    observation=(
                        f"Итоговый скор — среднее калиброванных вероятностей "
                        f"по {data.games_analyzed} партиям."
                    ),
                    interpretation=(
                        "Скор не заменяет ручную проверку и не является "
                        "доказательством нарушения."
                    ),
                    direction=ExplanationDirection.NEUTRAL,
                    contribution=None,
                )
            )
        return items

    def _limitations(self, data: ExplanationInput) -> list[str]:
        limitations = list(data.limitations)
        if not any(
            marker in limitation.casefold()
            for limitation in limitations
            for marker in _NO_PROOF_MARKERS
        ):
            limitations.append(
                "Автоматический анализ не является доказательством нарушения."
            )
        return limitations


def _direction_for(evidence: ExplanationEvidence) -> ExplanationDirection:
    if evidence.direction is not None:
        return evidence.direction
    if evidence.contribution is not None:
        if evidence.contribution > 0.0:
            return ExplanationDirection.SUPPORTS_RISK
        if evidence.contribution < 0.0:
            return ExplanationDirection.REDUCES_RISK
    return ExplanationDirection.NEUTRAL


def _contribution_text(evidence: ExplanationEvidence) -> str:
    if evidence.contribution is None:
        return f"{evidence.value:.2f} против базовой линии {evidence.baseline:.2f}"
    if evidence.contribution_share is None:
        return f"{evidence.contribution:+.3f}"
    return (
        f"{evidence.contribution:+.3f}, "
        f"доля {evidence.contribution_share:.0%}"
    )


def _interpretation(
    evidence: ExplanationEvidence,
    direction: ExplanationDirection,
) -> str:
    if direction is ExplanationDirection.SUPPORTS_RISK:
        return "Признак сдвигает решение модели в сторону повышенного риска."
    if direction is ExplanationDirection.REDUCES_RISK:
        return "Признак сдвигает решение модели в сторону снижения риска."
    return "Признак не меняет решение модели существенно."


__all__ = ["DETERMINISTIC_PROMPT_VERSION", "DeterministicExplainer"]
