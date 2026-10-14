"""Persistence-aware analysis, scoring, and verdict stages."""

from dataclasses import asdict
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from packages.chess_analysis import aggregate_game_features
from packages.chess_analysis.protocol import ChessEngine, EngineAnalysisError
from packages.llm import DeterministicExplainer, ExplanationEvidence, ExplanationInput
from packages.ml import BaselineScorer, ScoringInput

from ..db.models import (
    GameFeaturesModel,
    GameModel,
    GameScoreModel,
    ScanModel,
    UserModel,
    VerdictModel,
)
from .pipeline import ScanPipelineError


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
                    moves = await self._engine.analyze_game(
                        game.pgn or "", user.username_normalized
                    )
                    aggregate = aggregate_game_features(moves)
                    session.add(
                        GameFeaturesModel(
                            id=str(uuid4()),
                            game_id=game.id,
                            feature_version=self._feature_version,
                            aggregates=aggregate.model_dump(),
                            moves=[asdict(move) for move in moves],
                            engine_name=self._engine.config.engine_name,
                            engine_version=self._engine.config.engine_version,
                            depth_or_time_limit=f"{self._engine.config.depth}/{self._engine.config.time_ms}",
                        )
                    )
                scan.games_analyzed = len(games)
                await session.commit()
            except (EngineAnalysisError, ValueError) as exc:
                await session.rollback()
                raise ScanPipelineError("ANALYSIS_FAILED", str(exc)) from exc


class ScoringStage:
    """Score persisted aggregate features and store the numeric verdict."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        scorer: BaselineScorer,
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
                )
            ).all()
            if user is None:
                raise ScanPipelineError("USER_NOT_FOUND", "scan user not found")
            try:
                result = self._scorer.score(
                    ScoringInput(
                        username_normalized=user.username_normalized,
                        games=[row[1].aggregates for row in rows],
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ScanPipelineError("SCORING_FAILED", str(exc)) from exc
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
                    prompt_version="deterministic-v1",
                )
            )
            session.add(
                VerdictModel(
                    id=str(uuid4()),
                    scan_id=scan.id,
                    risk_level=result.risk_level,
                    risk_score=result.risk_score,
                    summary=explanation.summary,
                    evidence=[item.model_dump() for item in result.evidence],
                    limitations=explanation.limitations,
                    model_version=result.model.model_version,
                    prompt_version=explanation.prompt_version,
                )
            )
            await session.commit()
