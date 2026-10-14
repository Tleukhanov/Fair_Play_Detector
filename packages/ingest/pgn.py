"""Streaming PGN parsing with per-game error isolation."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import StringIO

import chess.pgn


class PgnParseError(ValueError):
    """Raised when one PGN game cannot be normalized."""


@dataclass(frozen=True, slots=True)
class ParsedGame:
    """Normalized metadata and original PGN for one valid game."""

    headers: dict[str, str]
    pgn: str

    @property
    def lichess_game_id(self) -> str:
        game_id = self.headers.get("GameId", "")
        if not game_id:
            raise PgnParseError("PGN is missing GameId")
        return game_id


def parse_pgn_game(raw_pgn: str) -> ParsedGame:
    """Parse and validate one game without hiding malformed input."""

    game = chess.pgn.read_game(StringIO(raw_pgn))
    if game is None or not game.headers:
        raise PgnParseError("PGN contains no game")
    if game.end().board().fullmove_number <= 1:
        raise PgnParseError("PGN contains no moves")
    headers = {str(key): str(value) for key, value in game.headers.items()}
    if headers.get("Variant", "Standard") != "Standard":
        raise PgnParseError("unsupported chess variant")
    if headers.get("Result") not in {"1-0", "0-1", "1/2-1/2", "*"}:
        raise PgnParseError("invalid game result")
    return ParsedGame(headers=headers, pgn=raw_pgn)


def iter_pgn_games(
    chunks: Iterable[bytes], errors: list[str] | None = None
) -> Iterator[ParsedGame]:
    """Parse complete games from byte chunks while retaining only one game."""

    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        parts = buffer.split("\n[Event ", 1)
        while len(parts) == 2:
            raw = parts[0].strip()
            buffer = "[Event " + parts[1]
            try:
                yield parse_pgn_game(raw)
            except PgnParseError as exc:
                if errors is not None:
                    errors.append(str(exc))
            parts = buffer.split("\n[Event ", 1)
    if buffer.strip():
        try:
            yield parse_pgn_game(buffer.strip())
        except PgnParseError as exc:
            if errors is not None:
                errors.append(str(exc))
