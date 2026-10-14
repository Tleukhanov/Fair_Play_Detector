"""Versioned scan API routes."""

from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.ingest.username import normalize_username

from ..db.models import GameModel, ScanModel, UserModel
from ..db.session import get_session
from ..schemas.api import (
    GameAggregateDTO,
    GameDetailResponse,
    GamesResponse,
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
        created_at=datetime.utcnow(),
    )
    session.add(scan)
    await session.commit()
    try:
        executor.submit(UUID(scan.id))
    except ExecutorCapacityError as exc:
        scan = await session.get(ScanModel, scan.id)
        if scan is not None:
            scan.status = ScanStatus.FAILED.value
            scan.error_code = "QUEUE_FULL"
            scan.error_message = "scan queue capacity exceeded"
            scan.completed_at = datetime.utcnow()
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
    raise HTTPException(status_code=404, detail="verdict not found")


@router.get("/{scan_id}/games", response_model=GamesResponse)
async def list_scan_games(
    scan_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> GamesResponse:
    """Return a paginated list of games belonging to a scan."""

    scan_exists = await session.scalar(select(ScanModel.id).where(ScanModel.id == str(scan_id)))
    if scan_exists is None:
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
    return GamesResponse(
        scan_id=scan_id,
        items=[_game_to_dto(game) for game in games],
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
    """Return game metadata; move details will be supplied by analysis workers."""

    game = await session.scalar(
        select(GameModel).where(
            GameModel.id == game_id,
            GameModel.scan_id == str(scan_id),
        )
    )
    if game is None:
        raise HTTPException(status_code=404, detail="game not found")
    return GameDetailResponse(
        scan_id=scan_id,
        game=_game_to_dto(game),
        moves=[],
    )


def _game_to_dto(game: GameModel) -> GameAggregateDTO:
    return GameAggregateDTO(
        game_id=game.lichess_game_id,
        source_url=game.source_url,
        played_at=game.played_at,
        speed=game.speed,
        result=game.result,
        user_color=game.user_color,
        move_count=0,
        top_move_match_rate=0.0,
        mean_centipawn_loss=0.0,
        maia_divergence=0.0,
    )
