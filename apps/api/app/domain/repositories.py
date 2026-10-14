"""Persistence ports; concrete database adapters belong to a later stage."""

from typing import Protocol
from uuid import UUID

from .models import Game, GameFeatures, GameScore, Scan, User, Verdict


class UserRepository(Protocol):
    async def get_by_normalized_username(self, username: str) -> User | None: ...

    async def save(self, user: User) -> User: ...


class ScanRepository(Protocol):
    async def get(self, scan_id: UUID) -> Scan | None: ...

    async def save(self, scan: Scan) -> Scan: ...


class GameRepository(Protocol):
    async def save(self, game: Game) -> Game: ...

    async def list_for_scan(self, scan_id: UUID, offset: int, limit: int) -> list[Game]: ...


class FeatureRepository(Protocol):
    async def save(self, features: GameFeatures) -> GameFeatures: ...

    async def get_latest_for_game(self, game_id: UUID) -> GameFeatures | None: ...


class ScoreRepository(Protocol):
    async def save(self, score: GameScore) -> GameScore: ...


class VerdictRepository(Protocol):
    async def get_for_scan(self, scan_id: UUID) -> Verdict | None: ...

    async def save(self, verdict: Verdict) -> Verdict: ...
