"""Safe explanation fallback when no LLM provider is configured."""

from .contracts import ExplanationInput, ExplanationOutput


class DeterministicExplainer:
    """Generate a grounded explanation without inventing facts."""

    def explain(self, data: ExplanationInput) -> ExplanationOutput:
        evidence_codes = ", ".join(item.code for item in data.evidence) or "none"
        summary = (
            f"Risk level: {data.risk_level}; score {data.risk_score:.2f} "
            f"based on {data.games_analyzed} analyzed games. "
            f"Evidence: {evidence_codes}. This is a probabilistic signal, not proof of cheating."
        )
        return ExplanationOutput(
            summary=summary,
            limitations=data.limitations,
            source="deterministic",
        )
