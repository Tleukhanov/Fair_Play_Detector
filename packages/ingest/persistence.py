"""Persistence adapter for streamed user-game ingestion."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db.models import GameModel, ScanModel

from .client import LichessClient
from .pgn import ParsedGame, iter_pgn_games


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Counts produced by one idempotent scan ingest."""

    games_found: int
    games_created: int
    games_skipped: int
    parse_errors: tuple[str, ...]


class LichessIngestService:
    """Fetch and persist public games while avoiding per-user duplicates."""

    def __init__(self, client: LichessClient) -> None:
        self._client = client

    async def ingest_scan(
        self,
        session: AsyncSession,
        scan: ScanModel,
        *,
        username: str,
        rated_only: bool = True,
    ) -> IngestResult:
        """Persist at most the scan limit and update its found-game count."""

        chunks = [
            chunk
            async for chunk in self._client.iter_pgn_chunks(
                username,
                max_games=scan.requested_game_limit,
                speed=scan.requested_time_control,
                rated_only=rated_only,
            )
        ]
        errors: list[str] = []
        parsed_games = list(iter_pgn_games(chunks, errors))
        game_ids = {game.lichess_game_id for game in parsed_games}
        existing_ids = set(
            (
                await session.scalars(
                    select(GameModel.lichess_game_id).where(
                        GameModel.user_id == scan.user_id,
                        GameModel.lichess_game_id.in_(game_ids),
                    )
                )
            ).all()
        )
        created = 0
        for parsed in parsed_games:
            if parsed.lichess_game_id in existing_ids:
                continue
            session.add(_to_game_model(parsed, scan, username))
            existing_ids.add(parsed.lichess_game_id)
            created += 1
        scan.games_found = len(parsed_games)
        await session.commit()
        return IngestResult(
            games_found=len(parsed_games),
            games_created=created,
            games_skipped=len(parsed_games) - created,
            parse_errors=tuple(errors),
        )


def _to_game_model(parsed: ParsedGame, scan: ScanModel, username: str) -> GameModel:
    headers = parsed.headers
    played_at = _played_at(headers)
    white = headers.get("White", "")
    black = headers.get("Black", "")
    user_color = "white" if white.casefold() == username else "black"
    game_id = parsed.lichess_game_id
    return GameModel(
        id=str(uuid4()),
        lichess_game_id=game_id,
        user_id=scan.user_id,
        scan_id=scan.id,
        played_at=played_at,
        rated=headers.get("Event", "").casefold().startswith("rated"),
        speed=headers.get("Speed", scan.requested_time_control),
        variant=headers.get("Variant", "Standard"),
        white_username=white,
        black_username=black,
        user_color=user_color,
        result=headers["Result"],
        pgn=parsed.pgn,
        pgn_hash=hashlib.sha256(parsed.pgn.encode("utf-8")).hexdigest(),
        source_url=f"https://lichess.org/{game_id}",
    )


def _played_at(headers: dict[str, str]) -> datetime:
    raw_date = headers.get("UTCDate", "1970.01.01").replace(".", "-")
    raw_time = headers.get("UTCTime", "00:00:00")
    return datetime.fromisoformat(f"{raw_date}T{raw_time}").replace(tzinfo=UTC).replace(
        tzinfo=None
    )
