"""End-to-end tests for the analysis -> scoring -> verdict pipeline.

Stockfish and the trained artifact are intentionally absent: the tests inject a
deterministic engine and the baseline scorer, and use an in-memory SQLite
database.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apps.api.app.api.routes import get_scan_game, list_scan_games
from apps.api.app.config import Settings
from apps.api.app.db.base import Base
from apps.api.app.db.models import (
    GameFeaturesModel,
    GameModel,
    GameScoreModel,
    ScanModel,
    UserModel,
    VerdictModel,
)
from apps.api.app.domain.models import DomainScanStatus
from apps.api.app.workers.analysis_stages import ChessAnalysisStage, ScoringStage
from apps.api.app.workers.scoring_factory import BASELINE_MODEL_VERSION, create_scorer
from apps.api.app.workers.store import ScanWorkerStore, reap_stale_scans
from packages.chess_analysis.models import AnalysisConfig, MoveAnalysis
from packages.ingest import LichessIngestService
from packages.llm import DeterministicExplainer
from packages.ml import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    BaselineScorer,
    ModelMetadata,
    RiskThresholds,
)

PGN = """[Event "rated blitz game"]
[Site "https://lichess.org/abcd1234"]
[GameId "abcd1234"]
[Variant "Standard"]
[Speed "blitz"]
[UTCDate "2026.01.02"]
[UTCTime "03:04:05"]
[White "Tester"]
[Black "Opponent"]
[Result "1-0"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""


class FakeChessEngine:
    """Deterministic engine that never launches a real binary."""

    def __init__(self, *, moves: int = 20, engine_version: str = "fixture-1.0") -> None:
        self._config = AnalysisConfig(
            engine_name="stockfish",
            engine_version=engine_version,
            depth=15,
            time_ms=100,
        )
        self._move_count = moves
        self.calls: list[tuple[str, str]] = []

    @property
    def config(self) -> AnalysisConfig:
        return self._config

    async def analyze_game(self, pgn: str, username: str) -> Sequence[MoveAnalysis]:
        self.calls.append((pgn, username))
        return [_fake_move(index) for index in range(self._move_count)]


def _fake_move(index: int) -> MoveAnalysis:
    """Build one mover-POV move record with a stable 50% top-1 rate."""

    is_top = index % 2 == 0
    return MoveAnalysis(
        ply=index * 2 + 1,
        fen_before=f"8/8/8/8/8/8/8/8 w - - 0 {index}",
        played_move_uci="e2e4",
        played_move_san="e4",
        eval_before_cp=30,
        eval_best_move_cp=30,
        eval_played_move_cp=10,
        is_top_1=is_top,
        is_top_3=True,
        time_seconds=1.5 + (index % 3),
        analysis_depth=15,
        is_forced=False,
        is_noisy_position=False,
        centipawn_loss=20.0,
        is_top1_sf1=is_top,
        is_top1_sf9=is_top,
        is_top1_sf15=is_top,
    )


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_version="baseline-fixture",
        model_type="baseline",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(FEATURE_NAMES),
        engine_name="stockfish",
        engine_version="fixture-1.0",
        analysis_depth=15,
        analysis_time_ms=100,
        thresholds=RiskThresholds(suspicious=0.55, high_risk=0.75),
    )


class MemoryDatabase:
    """In-memory async SQLite database with all tables created."""

    def __init__(self, engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]):
        self.engine = engine
        self.session_factory = session_factory

    async def close(self) -> None:
        await self.engine.dispose()


async def _new_database() -> MemoryDatabase:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return MemoryDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))


async def _seed_scan(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    games: int = 12,
    pgn_hashes: Sequence[str] | None = None,
) -> str:
    """Create one user, one fetching scan with a lease, and its games."""

    user = UserModel(
        id=str(uuid4()),
        lichess_username="Tester",
        username_normalized="tester",
    )
    scan = ScanModel(
        id=str(uuid4()),
        user_id=user.id,
        status=DomainScanStatus.FETCHING.value,
        requested_time_control="blitz",
        requested_game_limit=max(10, games),
        rated_only=True,
        lease_id=str(uuid4()),
        created_at=datetime.utcnow(),
    )
    async with session_factory() as session:
        session.add_all([user, scan])
        await session.flush()
        for index in range(games):
            pgn_hash = pgn_hashes[index] if pgn_hashes is not None else f"pgn-hash-{index}"
            session.add(
                GameModel(
                    id=str(uuid4()),
                    lichess_game_id=f"game{index}",
                    user_id=user.id,
                    scan_id=scan.id,
                    played_at=datetime(2026, 1, 1) + timedelta(minutes=index),
                    rated=True,
                    speed="blitz",
                    variant="Standard",
                    white_username="Tester",
                    black_username="Opponent",
                    user_color="white",
                    result="1-0",
                    pgn=PGN,
                    pgn_hash=pgn_hash,
                    source_url=f"https://lichess.org/game{index}",
                )
            )
        await session.commit()
    return scan.id


async def _lease(session_factory: async_sessionmaker[AsyncSession], scan_id: str) -> str:
    async with session_factory() as session:
        scan = await session.get(ScanModel, scan_id)
        assert scan is not None
        return str(scan.lease_id)


async def _count(session_factory: async_sessionmaker[AsyncSession], statement) -> int:
    async with session_factory() as session:
        return int(await session.scalar(statement) or 0)


def test_analysis_scoring_and_verdict_pipeline() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        scan_id = await _seed_scan(session_factory, games=12)
        engine = FakeChessEngine()
        store = ScanWorkerStore(session_factory)
        lease_id = await _lease(session_factory, scan_id)
        scan_uuid = UUID(scan_id)

        await store.transition(
            scan_uuid, lease_id, DomainScanStatus.FETCHING, DomainScanStatus.ANALYZING
        )
        await ChessAnalysisStage(session_factory, engine).run(scan_uuid)

        await store.transition(
            scan_uuid, lease_id, DomainScanStatus.ANALYZING, DomainScanStatus.SCORING
        )
        scoring = ScoringStage(
            session_factory,
            BaselineScorer(_metadata()),
            explainer=DeterministicExplainer(),
        )
        await scoring.run(scan_uuid)

        await store.transition(
            scan_uuid, lease_id, DomainScanStatus.SCORING, DomainScanStatus.EXPLAINING
        )
        await store.transition(
            scan_uuid, lease_id, DomainScanStatus.EXPLAINING, DomainScanStatus.COMPLETED
        )

        assert len(engine.calls) == 12
        assert (
            await _count(
                session_factory,
                select(func.count()).select_from(GameFeaturesModel),
            )
            == 12
        )
        assert (
            await _count(
                session_factory,
                select(func.count()).select_from(GameScoreModel),
            )
            == 12
        )
        assert (
            await _count(session_factory, select(func.count()).select_from(VerdictModel)) == 1
        )

        async with session_factory() as session:
            scan = await session.get(ScanModel, scan_id)
            assert scan is not None
            assert scan.status == DomainScanStatus.COMPLETED.value
            assert scan.completed_at is not None
            assert scan.model_version == "baseline-fixture"
            assert scan.games_analyzed == 12
            verdict = await session.scalar(
                select(VerdictModel).where(VerdictModel.scan_id == scan_id)
            )
            assert verdict is not None
            assert 0.0 <= verdict.risk_score <= 1.0
            assert verdict.confidence_lower <= verdict.risk_score <= verdict.confidence_upper
            assert verdict.evidence
            assert verdict.model_version == "baseline-fixture"

        # Re-running scoring must replace rows, not duplicate them.
        await scoring.run(scan_uuid)
        assert (
            await _count(
                session_factory,
                select(func.count()).select_from(GameScoreModel),
            )
            == 12
        )
        assert (
            await _count(session_factory, select(func.count()).select_from(VerdictModel)) == 1
        )

        await database.close()

    asyncio.run(scenario())


def test_cache_hit_copies_features_onto_the_current_game() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        scan_id = await _seed_scan(
            session_factory,
            games=2,
            pgn_hashes=["shared-pgn-hash", "shared-pgn-hash"],
        )
        engine = FakeChessEngine()

        await ChessAnalysisStage(session_factory, engine).run(UUID(scan_id))

        # Only the first game is analyzed; the second is served from the cache.
        assert len(engine.calls) == 1
        async with session_factory() as session:
            rows = (
                await session.scalars(
                    select(GameFeaturesModel).order_by(GameFeaturesModel.game_id)
                )
            ).all()
            assert len(rows) == 2
            assert len({row.id for row in rows}) == 2
            assert len({row.game_id for row in rows}) == 2
            assert rows[0].cache_key == rows[1].cache_key
            assert rows[0].aggregates == rows[1].aggregates
            assert len(rows[0].moves) == 20
            scan = await session.get(ScanModel, scan_id)
            assert scan is not None
            assert scan.games_analyzed == 2
        await database.close()

    asyncio.run(scenario())


def test_reanalysis_with_a_new_engine_version_replaces_feature_rows() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        scan_id = await _seed_scan(session_factory, games=10)
        scan_uuid = UUID(scan_id)

        await ChessAnalysisStage(session_factory, FakeChessEngine()).run(scan_uuid)
        upgraded = FakeChessEngine(engine_version="fixture-2.0")
        await ChessAnalysisStage(session_factory, upgraded).run(scan_uuid)

        # The stale analysis identity is replaced, not accumulated.
        assert len(upgraded.calls) == 10
        async with session_factory() as session:
            rows = (
                await session.scalars(select(GameFeaturesModel))
            ).all()
            assert len(rows) == 10
            assert {row.engine_version for row in rows} == {"fixture-2.0"}

        # Exactly one score row per game proves scoring stayed consistent.
        await ScoringStage(session_factory, BaselineScorer(_metadata())).run(scan_uuid)
        assert (
            await _count(
                session_factory,
                select(func.count()).select_from(GameScoreModel),
            )
            == 10
        )
        await database.close()

    asyncio.run(scenario())


class FakeLichessClient:
    """Streams one fixture PGN and records the export filters."""

    def __init__(self) -> None:
        self.rated_only_calls: list[bool] = []

    async def iter_pgn_chunks(
        self,
        _username: str,
        *,
        max_games: int = 30,
        speed: str = "blitz",
        rated_only: bool = True,
    ) -> AsyncIterator[bytes]:
        self.rated_only_calls.append(rated_only)
        yield PGN.encode()


def test_rescan_keeps_a_game_row_per_scan() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        client = FakeLichessClient()
        service = LichessIngestService(client)  # type: ignore[arg-type]

        async with session_factory() as session:
            user = UserModel(
                id=str(uuid4()),
                lichess_username="Tester",
                username_normalized="tester",
            )
            first = ScanModel(
                id=str(uuid4()),
                user=user,
                status="fetching",
                requested_time_control="blitz",
                requested_game_limit=10,
                rated_only=False,
            )
            session.add_all([user, first])
            await session.commit()
            user_id = user.id
            first_id = first.id

        async with session_factory() as session:
            scan = await session.get(ScanModel, first_id)
            assert scan is not None
            first_result = await service.ingest_scan(
                session, scan, username="tester", rated_only=scan.rated_only
            )
        assert first_result.games_created == 1
        assert client.rated_only_calls == [False]

        async with session_factory() as session:
            second = ScanModel(
                id=str(uuid4()),
                user_id=user_id,
                status="fetching",
                requested_time_control="blitz",
                requested_game_limit=10,
                rated_only=True,
            )
            session.add(second)
            await session.commit()
            second_id = second.id

        async with session_factory() as session:
            scan = await session.get(ScanModel, second_id)
            assert scan is not None
            second_result = await service.ingest_scan(
                session, scan, username="tester", rated_only=scan.rated_only
            )
        assert second_result.games_created == 1

        async with session_factory() as session:
            counts = {
                scan_id: await session.scalar(
                    select(func.count())
                    .select_from(GameModel)
                    .where(GameModel.scan_id == scan_id)
                )
                for scan_id in (first_id, second_id)
            }
        assert counts == {first_id: 1, second_id: 1}
        await database.close()

    asyncio.run(scenario())


def test_game_detail_accepts_public_and_internal_ids() -> None:
    """The games list exposes the Lichess id, so detail must resolve it too."""

    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        scan_id = await _seed_scan(session_factory, games=1)
        scan_uuid = UUID(scan_id)

        async with session_factory() as session:
            game = await session.scalar(
                select(GameModel).where(GameModel.scan_id == scan_id)
            )
            assert game is not None
            assert game.lichess_game_id == "game0"
            assert game.id != game.lichess_game_id

            by_public_id = await get_scan_game(
                scan_uuid, game.lichess_game_id, session=session
            )
            by_internal_id = await get_scan_game(scan_uuid, game.id, session=session)

            assert by_public_id.game.game_id == "game0"
            assert by_internal_id.game.game_id == "game0"
            assert by_public_id.game.game_id == by_internal_id.game.game_id

        await database.close()

    asyncio.run(scenario())


def test_game_dto_and_detail_expose_persisted_analysis() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        scan_id = await _seed_scan(session_factory, games=10)
        scan_uuid = UUID(scan_id)

        await ChessAnalysisStage(session_factory, FakeChessEngine()).run(scan_uuid)
        await ScoringStage(session_factory, BaselineScorer(_metadata())).run(scan_uuid)

        async with session_factory() as session:
            listing = await list_scan_games(scan_uuid, page=1, page_size=20, session=session)
            assert listing.total == 10
            assert len(listing.items) == 10
            item = listing.items[0]
            assert item.move_count == 20
            assert item.top_move_match_rate == pytest.approx(0.5)
            assert item.mean_centipawn_loss == pytest.approx(20.0)
            assert item.maia_divergence is None

            game = await session.scalar(
                select(GameModel).where(GameModel.scan_id == scan_id).limit(1)
            )
            assert game is not None
            detail = await get_scan_game(scan_uuid, game.id, session=session)
            assert detail.game.game_id == game.lichess_game_id
            assert detail.game.move_count == 20
            assert len(detail.moves) == 20
            move = detail.moves[0]
            assert move.ply == 1
            assert move.fen_before.startswith("8/8/8/8")
            assert move.played_move_uci == "e2e4"
            assert move.centipawn_loss == pytest.approx(20.0)
            assert move.is_top_1 is True
            assert move.is_top_3 is True
            assert move.time_seconds == pytest.approx(1.5)
            assert move.maia_probability is None
        await database.close()

    asyncio.run(scenario())


def test_reap_stale_scans_fails_only_old_non_terminal_scans() -> None:
    async def scenario() -> None:
        database = await _new_database()
        session_factory = database.session_factory
        now = datetime.utcnow()
        user = UserModel(
            id=str(uuid4()),
            lichess_username="Tester",
            username_normalized="tester",
        )
        stale = ScanModel(
            id=str(uuid4()),
            user_id=user.id,
            status=DomainScanStatus.FETCHING.value,
            requested_time_control="blitz",
            requested_game_limit=10,
            created_at=now - timedelta(hours=3),
            started_at=now - timedelta(minutes=120),
        )
        fresh = ScanModel(
            id=str(uuid4()),
            user_id=user.id,
            status=DomainScanStatus.ANALYZING.value,
            requested_time_control="blitz",
            requested_game_limit=10,
            created_at=now - timedelta(minutes=10),
            started_at=now - timedelta(minutes=5),
        )
        finished = ScanModel(
            id=str(uuid4()),
            user_id=user.id,
            status=DomainScanStatus.COMPLETED.value,
            requested_time_control="blitz",
            requested_game_limit=10,
            created_at=now - timedelta(hours=3),
            started_at=now - timedelta(hours=3),
            completed_at=now - timedelta(hours=2),
        )
        async with session_factory() as session:
            session.add(user)
            session.add_all([stale, fresh, finished])
            await session.commit()

        assert await reap_stale_scans(session_factory, 60) == 1

        async with session_factory() as session:
            stale_row = await session.get(ScanModel, stale.id)
            fresh_row = await session.get(ScanModel, fresh.id)
            finished_row = await session.get(ScanModel, finished.id)
            assert stale_row is not None
            assert stale_row.status == DomainScanStatus.FAILED.value
            assert stale_row.error_code == "STALE_SCAN"
            assert stale_row.completed_at is not None
            assert fresh_row is not None
            assert fresh_row.status == DomainScanStatus.ANALYZING.value
            assert finished_row is not None
            assert finished_row.status == DomainScanStatus.COMPLETED.value
        await database.close()

    asyncio.run(scenario())


def test_scoring_factory_uses_baseline_without_artifact(tmp_path) -> None:
    settings = Settings(model_dir=str(tmp_path), stockfish_path="stockfish")
    scorer = create_scorer(settings)
    assert isinstance(scorer, BaselineScorer)
    assert BASELINE_MODEL_VERSION == "baseline-heuristic-v1"
