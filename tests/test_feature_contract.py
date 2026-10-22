"""Regression tests for the canonical feature contract."""

import pytest

from packages.ml.feature_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FORBIDDEN_COLUMNS,
    FeatureContractError,
    Phase,
    validate_feature_columns,
    validate_feature_values,
    validate_player_disjoint,
)


def test_feature_order_is_stable() -> None:
    assert FEATURE_NAMES[0] == "centipawn_loss"
    assert FEATURE_NAMES[-1] == "opponent_elo_centered"
    assert len(FEATURE_NAMES) == 12
    assert FEATURE_SCHEMA_VERSION == "2.0.0"


def test_contract_accepts_exact_columns() -> None:
    validate_feature_columns(FEATURE_NAMES)


@pytest.mark.parametrize("leaked", sorted(FORBIDDEN_COLUMNS)[:10])
def test_contract_rejects_each_forbidden_column(leaked: str) -> None:
    columns = list(FEATURE_NAMES) + [leaked]
    with pytest.raises(FeatureContractError, match="unexpected|forbidden"):
        validate_feature_columns(columns)


def test_contract_rejects_missing_column() -> None:
    with pytest.raises(FeatureContractError, match="missing"):
        validate_feature_columns(FEATURE_NAMES[:-1])


def test_contract_rejects_reordered_columns() -> None:
    reordered = (FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:])
    with pytest.raises(FeatureContractError, match="order"):
        validate_feature_columns(reordered)


def test_contract_rejects_duplicate_column() -> None:
    with pytest.raises(FeatureContractError, match="duplicate"):
        validate_feature_columns((*FEATURE_NAMES, FEATURE_NAMES[0]))


def test_player_disjoint_guard() -> None:
    validate_player_disjoint({"a", "b"}, {"c", "d"})
    with pytest.raises(FeatureContractError, match="not player-disjoint"):
        validate_player_disjoint({"a", "b"}, {"b", "c"})


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in FEATURE_NAMES}
    row["phase"] = int(Phase.MIDDLEGAME)
    row.update(overrides)
    return row


def test_feature_values_accept_clean_row() -> None:
    validate_feature_values([_row(), _row(phase=int(Phase.ENDGAME))])


def test_feature_values_reject_nan_and_phase() -> None:
    with pytest.raises(FeatureContractError, match="not finite"):
        validate_feature_values([_row(match_sf15=float("nan"))])
    with pytest.raises(FeatureContractError, match="unknown phase"):
        validate_feature_values([_row(phase=99)])
    with pytest.raises(FeatureContractError, match="does not match"):
        validate_feature_values([{"centipawn_loss": 1.0}])
