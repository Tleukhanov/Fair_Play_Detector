"""Versioned scan API routes."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import HttpUrl
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.chess_analysis.models import EVAL_CLIP_CP
from packages.ingest.username import normalize_username

from ..db.models import (
    GameFeaturesModel,
    GameModel,
    GameScoreModel,
    ScanModel,
    UserModel,
    VerdictModel,
)
from ..db.session import get_session
from ..schemas.api import (
    EvidenceItem,
    GameAggregateDTO,
    GameDetailResponse,
    GameMoveDTO,
    GamesResponse,
    RiskLevel,
    ScanCreateRequest,
    ScanCreateResponse,
    ScanStatus,
    ScanStatusResponse,
    VerdictResponse,
)
from ..workers.dependencies import get_scan_executor
from ..workers.executor import BackgroundScanExecutor, ExecutorCapacityError

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])


@router.post("", response_model=ScanCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_scan(
    request: ScanCreateRequest,
    session: AsyncSession = Depends(get_session),
    executor: BackgroundScanExecutor = Depends(get_scan_executor),
) -> ScanCreateResponse:
    """Create a queued scan after validating and normalizing its username."""

    username = normalize_username(request.username)
    user = await session.scalar(
        select(UserModel).where(UserModel.username_normalized == username)
    )
    if user is None:
        user = UserModel(
            id=str(uuid4()),
            lichess_username=request.username.strip(),
            username_normalized=username,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(user)
        await session.flush()

    scan = ScanModel(
        id=str(uuid4()),
        user_id=user.id,
        status=ScanStatus.QUEUED.value,
        requested_time_control=request.speed,
        requested_game_limit=request.max_games,
        rated_only=request.rated_only,
        created_at=datetime.utcnow(),
    )
    session.add(scan)
    await session.commit()
    try:
        executor.submit(UUID(scan.id))
    except ExecutorCapacityError as exc:
        failed_scan = await session.get(ScanModel, scan.id)
        if failed_scan is not None:
            failed_scan.status = ScanStatus.FAILED.value
            failed_scan.error_code = "QUEUE_FULL"
            failed_scan.error_message = "scan queue capacity exceeded"
            failed_scan.completed_at = datetime.utcnow()
            await session.commit()
        raise HTTPException(status_code=503, detail="scan queue capacity exceeded") from exc
    return ScanCreateResponse(
        scan_id=UUID(scan.id),
        status=ScanStatus.QUEUED,
        poll_url=f"/api/v1/scans/{scan.id}",
    )


@router.get("/{scan_id}", response_model=ScanStatusResponse)
async def get_scan_status(
    scan_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ScanStatusResponse:
    """Return current scan progress or an explicit not-found error."""

    scan = await session.get(ScanModel, str(scan_id))
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return ScanStatusResponse(
        scan_id=scan_id,
        status=ScanStatus(scan.status),
        games_found=scan.games_found,
        games_analyzed=scan.games_analyzed,
        current_stage=scan.status,
        error_code=scan.error_code,
        error_message=scan.error_message,
    )


@router.get("/{scan_id}/verdict", response_model=VerdictResponse)
async def get_scan_verdict(
    scan_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> VerdictResponse:
    """Return a completed verdict, or explain why it is not available yet."""

    scan = await session.get(ScanModel, str(scan_id))
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    if scan.status != ScanStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail="verdict is not available")
    verdict = await session.scalar(select(VerdictModel).where(VerdictModel.scan_id == str(scan_id)))
    user = await session.get(UserModel, scan.user_id)
    if verdict is None or user is None:
        raise HTTPException(status_code=404, detail="verdict not found")
    evidence = [EvidenceItem.model_validate(item) for item in verdict.evidence]
    confidence_interval = (verdict.confidence_lower, verdict.confidence_upper)
    return VerdictResponse(
        scan_id=scan_id,
        username=user.lichess_username,
        risk_level=RiskLevel(verdict.risk_level),
        risk_score=verdict.risk_score,
        confidence_interval=confidence_interval,
        games_analyzed=scan.games_analyzed,
        summary=verdict.summary,
        evidence=evidence,
        limitations=verdict.limitations,
        model_version=verdict.model_version,
    )


@router.get("/{scan_id}/games", response_model=GamesResponse)
async def list_scan_games(
    scan_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> GamesResponse:
    """Return a paginated list of games belonging to a scan."""

    scan = await session.get(ScanModel, str(scan_id))
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    total = await session.scalar(
        select(func.count()).select_from(GameModel).where(GameModel.scan_id == str(scan_id))
    )
    games = (
        await session.scalars(
            select(GameModel)
            .where(GameModel.scan_id == str(scan_id))
            .order_by(GameModel.played_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    game_ids = [game.id for game in games]
    # Two bulk queries instead of two per game.
    features = await _features_by_game(session, game_ids)
    scores = await _scores_by_game(session, game_ids, scan.model_version)
    return GamesResponse(
        scan_id=scan_id,
        items=[
            _game_to_dto(game, features.get(game.id), scores.get(game.id))
            for game in games
        ],
        page=page,
        page_size=page_size,
        total=total or 0,
    )


@router.get("/{scan_id}/games/{game_id}", response_model=GameDetailResponse)
async def get_scan_game(
    scan_id: UUID,
    game_id: str,
    session: AsyncSession = Depends(get_session),
) -> GameDetailResponse:
    """Return game metadata and the analyzed moves for that game."""

    scan = await session.get(ScanModel, str(scan_id))
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    game = await session.scalar(
        select(GameModel).where(
            GameModel.scan_id == str(scan_id),
            or_(GameModel.id == game_id, GameModel.lichess_game_id == game_id),
        )
    )
    if game is None:
        raise HTTPException(status_code=404, detail="game not found")
    features = await _features_by_game(session, [game.id])
    scores = await _scores_by_game(session, [game.id], scan.model_version)
    game_features = features.get(game.id)
    return GameDetailResponse(
        scan_id=scan_id,
        game=_game_to_dto(game, game_features, scores.get(game.id)),
        moves=_moves_to_dto(game_features),
    )


async def _features_by_game(
    session: AsyncSession,
    game_ids: Sequence[str],
) -> dict[str, GameFeaturesModel]:
    """Return the newest feature row per game in one query."""

    if not game_ids:
        return {}
    rows = (
        await session.scalars(
            select(GameFeaturesModel)
            .where(GameFeaturesModel.game_id.in_(game_ids))
            .order_by(GameFeaturesModel.created_at)
        )
    ).all()
    latest: dict[str, GameFeaturesModel] = {}
    for row in rows:
        latest[row.game_id] = row
    return latest


async def _scores_by_game(
    session: AsyncSession,
    game_ids: Sequence[str],
    model_version: str | None,
) -> dict[str, GameScoreModel]:
    """Return one score row per game, preferring the scan's model version."""

    if not game_ids:
        return {}
    rows = (
        await session.scalars(
            select(GameScoreModel).where(GameScoreModel.game_id.in_(game_ids))
        )
    ).all()
    indexed: dict[str, GameScoreModel] = {}
    for row in rows:
        existing = indexed.get(row.game_id)
        if existing is None or row.model_version == model_version:
            indexed[row.game_id] = row
    return indexed


def _game_to_dto(
    game: GameModel,
    features: GameFeaturesModel | None = None,
    score: GameScoreModel | None = None,
) -> GameAggregateDTO:
    """Map a game plus its persisted analysis into the aggregate DTO."""

    aggregates: Mapping[str, Any] = features.aggregates if features is not None else {}
    move_count_value = _as_float(aggregates.get("move_count"))
    move_count = max(0, int(move_count_value)) if move_count_value is not None else 0
    top_move_match_rate = _clamp_rate(_as_float(aggregates.get("top_1_match_rate")))
    mean_centipawn_loss = max(0.0, _as_float(aggregates.get("mean_centipawn_loss")) or 0.0)
    maia_divergence = _as_optional_non_negative(aggregates.get("maia_divergence"))
    if score is not None:
        top_move_match_rate = _clamp_rate(score.top_move_match_rate)
        mean_centipawn_loss = max(0.0, score.mean_centipawn_loss)
        if score.maia_divergence is not None:
            maia_divergence = max(0.0, score.maia_divergence)
    return GameAggregateDTO(
        game_id=game.lichess_game_id,
        source_url=cast(HttpUrl, game.source_url),
        played_at=game.played_at,
        speed=game.speed,
        result=game.result,
        user_color=game.user_color,
        move_count=move_count,
        top_move_match_rate=top_move_match_rate,
        mean_centipawn_loss=mean_centipawn_loss,
        maia_divergence=maia_divergence,
    )


def _moves_to_dto(features: GameFeaturesModel | None) -> list[GameMoveDTO]:
    """Map persisted move records, ignoring rows with unknown fields."""

    if features is None or not features.moves:
        return []
    moves: list[GameMoveDTO] = []
    for raw in features.moves:
        if not isinstance(raw, Mapping):
            continue
        move = _move_to_dto(raw)
        if move is not None:
            moves.append(move)
    return moves


def _move_to_dto(raw: Mapping[str, Any]) -> GameMoveDTO | None:
    ply = raw.get("ply")
    fen_before = raw.get("fen_before")
    played_move_uci = raw.get("played_move_uci")
    if isinstance(ply, bool) or not isinstance(ply, int) or ply < 1:
        return None
    if not isinstance(fen_before, str) or not isinstance(played_move_uci, str):
        return None
    time_seconds = _as_optional_non_negative(raw.get("time_seconds"))
    maia_probability = _as_optional_non_negative(raw.get("maia_move_probability"))
    if maia_probability is not None:
        maia_probability = min(1.0, maia_probability)
    played_move_san = raw.get("played_move_san")
    return GameMoveDTO(
        ply=ply,
        fen_before=fen_before,
        played_move_uci=played_move_uci,
        played_move_san=played_move_san if isinstance(played_move_san, str) else None,
        centipawn_loss=_move_centipawn_loss(raw),
        is_top_1=bool(raw.get("is_top_1") or raw.get("is_top1_sf15")),
        is_top_3=bool(raw.get("is_top_3")),
        time_seconds=time_seconds,
        maia_probability=maia_probability,
    )


def _move_centipawn_loss(raw: Mapping[str, Any]) -> float:
    value = _as_float(raw.get("centipawn_loss"))
    if value is None:
        before = _as_float(raw.get("eval_before_cp"))
        played = _as_float(raw.get("eval_played_move_cp"))
        value = before - played if before is not None and played is not None else 0.0
    return min(EVAL_CLIP_CP, max(0.0, value))


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_optional_non_negative(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return max(0.0, number)


def _clamp_rate(value: float | None) -> float:
    if value is None:
        return 0.0
    return min(1.0, max(0.0, value))
