"""Scoring strategy protocol shared by the baseline and trained models.

The API worker depends on this protocol, not on a concrete model.  That keeps
the stage layer stable while the ML layer evolves from the heuristic baseline to
a persisted, calibrated artifact.
"""

from __future__ import annotations

from typing import Protocol

from .contracts import ScoreResult, ScoringInput


class Scorer(Protocol):
    """A numeric scorer that turns aggregated games into a risk result."""

    def score(self, scoring_input: ScoringInput) -> ScoreResult: ...


__all__ = ["Scorer"]
