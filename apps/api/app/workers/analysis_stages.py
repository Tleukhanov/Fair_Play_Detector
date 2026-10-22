"""Persistence-aware analysis, scoring, and verdict stages."""

from dataclasses import asdict
from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from packages.chess_analysis import aggregate_game_features
from packages.chess_analysis.protocol import ChessEngine, EngineAnalysisError
from packages.feature_engineering.schema import GameAggregateFeatures
from packages.llm import DeterministicExplainer, ExplanationEvidence, ExplanationInput
from packages.ml import ScoringInput
from packages.ml.protocols import Scorer

from ..db.models import (
    GameFeaturesModel,
    GameModel,
    GameScoreModel,
    ScanModel,
    UserModel,
    VerdictModel,
)
from .pipeline import ScanPipelineError

#: Version stored with every generated explanation.
PROMPT_VERSION = "deterministic-v1"


class ChessAnalysisStage:
    """Analyze every persisted game with an injected bounded chess engine."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: ChessEngine,
        *,
        feature_version: str = "1.0.0",
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self._feature_version = feature_version

    async def run(self, scan_id: UUID) -> None:
        async with self._session_factory() as session:
            scan = await session.get(ScanModel, str(scan_id))
            if scan is None:
                raise ScanPipelineError("SCAN_NOT_FOUND", "scan not found")
            user = await session.get(UserModel, scan.user_id)
            games = (
                await session.scalars(select(GameModel).where(GameModel.scan_id == str(scan_id)))
            ).all()
            if user is None:
                raise ScanPipelineError("USER_NOT_FOUND", "scan user not found")
            try:
                for game in games:
                    cache_key = _analysis_cache_key(
                        game.pgn_hash,
                        self._engine.config.engine_name,
                        self._engine.config.engine_version,
                        self._engine.config.depth,
                        self._engine.config.time_ms,
                        self._feature_version,
                    )
                    already_analyzed = await session.scalar(
                        select(GameFeaturesModel.id).where(
                            GameFeaturesModel.game_id == game.id,
                            GameFeaturesModel.cache_key == cache_key,
                        )
                    )
                    if already_analyzed is not None:
                        # Idempotent re-run of the same analysis identity.
                        continue
                    # A different engine/feature identity invalidates the stale
                    # row: keep exactly one feature row per game so scoring
                    # cannot insert two scores for the same model version.
                    await _delete_game_features(session, game.id)
                    cached = await session.scalar(
                        select(GameFeaturesModel)
                        .where(GameFeaturesModel.cache_key == cache_key)
                        .order_by(GameFeaturesModel.created_at)
                        .limit(1)
                    )
                    if cached is not None:
                        # Reuse the expensive analysis, but keep a per-game row.
                        session.add(
                            _copy_features(cached, game_id=game.id, cache_key=cache_key)
                        )
                        continue
                    moves = await self._engine.analyze_game(
                        game.pgn or "", user.username_normalized
                    )
                    aggregate = aggregate_game_features(moves)
                    session.add(
                        GameFeaturesModel(
                            id=str(uuid4()),
                            game_id=game.id,
                            cache_key=cache_key,
                            pgn_hash=game.pgn_hash,
                            feature_version=self._feature_version,
                            aggregates=aggregate.model_dump(),
                            moves=[asdict(move) for move in moves],
                            engine_name=self._engine.config.engine_name,
                            engine_version=self._engine.config.engine_version,
                            depth_or_time_limit=f"{self._engine.config.depth}/{self._engine.config.time_ms}",
                        )
                    )
                await session.flush()
                analyzed = await session.scalar(
                    select(func.count(GameFeaturesModel.id))
                    .join(GameModel, GameModel.id == GameFeaturesModel.game_id)
                    .where(GameModel.scan_id == str(scan_id))
                )
                scan.games_analyzed = int(analyzed or 0)
                await session.commit()
            except (EngineAnalysisError, ValueError) as exc:
                await session.rollback()
                raise ScanPipelineError("ANALYSIS_FAILED", str(exc)) from exc


async def _delete_game_features(
    session: AsyncSession,
    game_id: str,
) -> None:
    """Remove stale analysis rows for one game before re-analysis."""

    rows = (
        await session.scalars(
            select(GameFeaturesModel).where(GameFeaturesModel.game_id == game_id)
        )
    ).all()
    for row in rows:
        await session.delete(row)


def _copy_features(
    cached: GameFeaturesModel,
    *,
    game_id: str,
    cache_key: str,
) -> GameFeaturesModel:
    """Clone cached analysis onto the current game with a fresh identity."""

    return GameFeaturesModel(
        id=str(uuid4()),
        game_id=game_id,
        cache_key=cache_key,
        pgn_hash=cached.pgn_hash,
        feature_version=cached.feature_version,
        aggregates=dict(cached.aggregates),
        moves=list(cached.moves),
        engine_name=cached.engine_name,
        engine_version=cached.engine_version,
        depth_or_time_limit=cached.depth_or_time_limit,
    )


def _analysis_cache_key(
    pgn_hash: str,
    engine_name: str,
    engine_version: str,
    depth: int,
    time_ms: int,
    feature_version: str,
) -> str:
    payload = "|".join(
        (pgn_hash, engine_name, engine_version, str(depth), str(time_ms), feature_version)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class ScoringStage:
    """Score persisted aggregate features and store the numeric verdict."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        scorer: Scorer,
        *,
        explainer: DeterministicExplainer | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._scorer = scorer
        self._explainer = explainer or DeterministicExplainer()

    async def run(self, scan_id: UUID) -> None:
        async with self._session_factory() as session:
            scan = await session.get(ScanModel, str(scan_id))
            if scan is None:
                raise ScanPipelineError("SCAN_NOT_FOUND", "scan not found")
            user = await session.get(UserModel, scan.user_id)
            rows = (
                await session.execute(
                    select(GameModel, GameFeaturesModel)
                    .join(GameFeaturesModel, GameFeaturesModel.game_id == GameModel.id)
                    .where(GameModel.scan_id == str(scan_id))
                    .order_by(GameModel.played_at)
                )
            ).all()
            if user is None:
                raise ScanPipelineError("USER_NOT_FOUND", "scan user not found")
            try:
                result = self._scorer.score(
                    ScoringInput(
                        username_normalized=user.username_normalized,
                        games=[
                            GameAggregateFeatures.model_validate(features.aggregates)
                            for _game, features in rows
                        ],
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ScanPipelineError("SCORING_FAILED", str(exc)) from exc

            # A re-run must replace earlier scores and the earlier verdict.
            scan_game_ids = select(GameModel.id).where(GameModel.scan_id == str(scan_id))
            await session.execute(
                delete(GameScoreModel).where(GameScoreModel.game_id.in_(scan_game_ids))
            )
            await session.execute(delete(VerdictModel).where(VerdictModel.scan_id == scan.id))
            for game, features in rows:
                session.add(
                    GameScoreModel(
                        id=str(uuid4()),
                        game_id=game.id,
                        model_version=result.model.model_version,
                        risk_score=result.risk_score,
                        top_move_match_rate=features.aggregates["top_1_match_rate"],
                        mean_centipawn_loss=features.aggregates["mean_centipawn_loss"],
                        maia_divergence=features.aggregates.get("maia_divergence"),
                        explainability={"risk_level": result.risk_level},
                    )
                )
            explanation = self._explainer.explain(
                ExplanationInput(
                    risk_level=result.risk_level,
                    risk_score=result.risk_score,
                    confidence_interval=result.confidence_interval,
                    games_analyzed=result.games_analyzed,
                    evidence=[
                        ExplanationEvidence(
                            code=item.code,
                            label=item.label,
                            value=item.value,
                            baseline=item.baseline,
                            weight=item.weight,
                        )
                        for item in result.evidence
                    ],
                    limitations=["Automated analysis is not proof of cheating."],
                    model_version=result.model.model_version,
                    prompt_version=PROMPT_VERSION,
                )
            )
            session.add(
                VerdictModel(
                    id=str(uuid4()),
                    scan_id=scan.id,
                    risk_level=result.risk_level,
                    risk_score=result.risk_score,
                    confidence_lower=result.confidence_interval[0],
                    confidence_upper=result.confidence_interval[1],
                    summary=explanation.summary,
                    evidence=[item.model_dump() for item in result.evidence],
                    limitations=explanation.limitations,
                    model_version=result.model.model_version,
                    prompt_version=PROMPT_VERSION,
                )
            )
            scan.model_version = result.model.model_version
            scan.games_analyzed = len(rows)
            await session.commit()
