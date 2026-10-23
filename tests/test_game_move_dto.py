"""GameMoveDTO mapping keeps the SAN move when the analyzer stored it."""

from apps.api.app.api.routes import _move_to_dto


def _raw_move() -> dict[str, object]:
    return {
        "ply": 12,
        "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "played_move_uci": "e2e4",
        "played_move_san": "e4",
        "centipawn_loss": 14.0,
        "is_top_1": True,
        "is_top_3": True,
        "time_seconds": 4.2,
        "maia_move_probability": 0.31,
    }


def test_move_to_dto_preserves_san() -> None:
    move = _move_to_dto(_raw_move())

    assert move is not None
    assert move.ply == 12
    assert move.played_move_uci == "e2e4"
    assert move.played_move_san == "e4"
    assert move.centipawn_loss == 14.0
    assert move.is_top_1 is True
    assert move.is_top_3 is True
    assert move.time_seconds == 4.2
    assert move.maia_probability == 0.31


def test_move_to_dto_allows_missing_san() -> None:
    raw = _raw_move()
    raw.pop("played_move_san")

    move = _move_to_dto(raw)

    assert move is not None
    assert move.played_move_san is None


def test_move_to_dto_ignores_non_string_san() -> None:
    raw = _raw_move()
    raw["played_move_san"] = 42

    move = _move_to_dto(raw)

    assert move is not None
    assert move.played_move_san is None
