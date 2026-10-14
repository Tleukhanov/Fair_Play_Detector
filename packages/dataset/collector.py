"""Safe, bounded dataset collection with an explicit post-collection gate."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import httpx

from packages.ingest.pgn import ParsedGame, iter_pgn_games

from .contracts import DatasetCollectionReport, DatasetFeatureRow, SourceCollectionReport

QUOTA_BYTES = 5 * 1024**3
FEATURE_COLUMNS = list(DatasetFeatureRow.model_fields)


class MissingCollectionPeriod(ValueError):
    """Raised before network access when a required period is missing."""


class DatasetQuotaExceeded(RuntimeError):
    """Raised immediately when the combined compressed byte quota is reached."""


class ByteQuota:
    """Count bytes received across all sources, including compressed bytes."""

    def __init__(self, limit: int = QUOTA_BYTES) -> None:
        self.limit = limit
        self.used = 0

    def consume(self, amount: int) -> None:
        if amount < 0 or self.used + amount > self.limit:
            raise DatasetQuotaExceeded("DATASET_QUOTA_EXCEEDED")
        self.used += amount


def collect_dataset(
    *,
    lichess_elite_month: str | None,
    tcec_season: str | None,
    output_dir: Path,
    lichess_url_template: str = "https://database.lichess.org/lichess_elite_{period}.pgn.zst",
    tcec_url_template: str = "https://www.open-chess.org/tcec/{period}.pgn.zst",
    quota_bytes: int = QUOTA_BYTES,
    client: httpx.Client | None = None,
) -> DatasetCollectionReport:
    """Collect both bounded sources and stop after writing the report."""

    _require_period("LICHESS_ELITE_MONTH", lichess_elite_month)
    _require_period("TCEC_SEASON", tcec_season)
    output_dir.mkdir(parents=True, exist_ok=True)
    quota = ByteQuota(quota_bytes)
    own_client = client is None
    http_client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    temporary_files: list[Path] = []
    reports: list[SourceCollectionReport] = []
    rows_by_class: dict[str, list[DatasetFeatureRow]] = {}
    try:
        for dataset_class, source, period, template in (
            ("human", "lichess_elite", lichess_elite_month, lichess_url_template),
            ("engine_like", "tcec", tcec_season, tcec_url_template),
        ):
            url = template.format(period=period)
            report, rows, temporary = _collect_source(
                http_client, quota, dataset_class, source, period, url
            )
            reports.append(report)
            rows_by_class[dataset_class] = rows
            temporary_files.append(temporary)
        parquet_files = {
            dataset_class: _write_parquet_atomic(
                output_dir / f"{'lichess_elite_' if dataset_class == 'human' else 'tcec_'}"
                f"{period}_features.parquet",
                rows,
            )
            for dataset_class, rows, period in (
                ("human", rows_by_class["human"], lichess_elite_month),
                ("engine_like", rows_by_class["engine_like"], tcec_season),
            )
        }
        all_rows = rows_by_class["human"] + rows_by_class["engine_like"]
        if len(all_rows) < 5:
            raise ValueError("dataset must contain at least five feature rows for the gate report")
        counts = {key: len(value) for key, value in rows_by_class.items()}
        total = len(all_rows)
        report = DatasetCollectionReport(
            lichess_elite_month=lichess_elite_month,
            tcec_season=tcec_season,
            quota_bytes=quota.limit,
            downloaded_bytes=quota.used,
            parquet_files=parquet_files,
            sources=reports,
            class_counts=counts,
            class_percentages={
                key: round(value / total * 100, 2) for key, value in counts.items()
            },
            sample_rows=all_rows[:5],
            columns=FEATURE_COLUMNS,
        )
        (output_dir / "collection_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
        if own_client:
            http_client.close()


def _collect_source(
    client: httpx.Client,
    quota: ByteQuota,
    dataset_class: str,
    source: str,
    period: str,
    url: str,
) -> tuple[SourceCollectionReport, list[DatasetFeatureRow], Path]:
    source_start = quota.used
    with client.stream("GET", url, headers={"Accept": "application/octet-stream"}) as response:
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix or ".archive"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
            temporary = Path(raw.name)
            for chunk in response.iter_bytes():
                quota.consume(len(chunk))
                raw.write(chunk)
    rows: list[DatasetFeatureRow] = []
    skip_reasons: Counter[str] = Counter()
    games_read = games_skipped = 0
    try:
        for game in _iter_archive_games(temporary):
            games_read += 1
            try:
                rows.append(_to_feature_row(game, dataset_class))
            except ValueError as exc:
                games_skipped += 1
                skip_reasons[str(exc)] += 1
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"dataset source format failure: {source}: {exc}") from exc
    return (
        SourceCollectionReport(
            source=source,
            period=period,
            url=url,
            downloaded_bytes=quota.used - source_start,
            games_read=games_read,
            games_processed=len(rows),
            games_skipped=games_skipped,
            skip_reasons=dict(skip_reasons),
        ),
        rows,
        temporary,
    )


def _iter_archive_games(path: Path) -> Iterator[ParsedGame]:
    opener = _decompressor(path)
    with opener(path, "rb") as stream:
        yield from iter_pgn_games(_chunks(stream))


def _decompressor(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open
    if suffix == ".bz2":
        return bz2.open
    if suffix in {".xz", ".lzma"}:
        return lzma.open
    if suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstandard is required for .zst dataset sources") from exc

        def open_zstd(source: Path, mode: str):
            if mode != "rb":
                raise ValueError("dataset zstd reader is read-only")
            handle = source.open("rb")
            return zstandard.ZstdDecompressor().stream_reader(handle)

        return open_zstd
    return open


def _chunks(stream: BinaryIO, size: int = 1024 * 1024) -> Iterator[bytes]:
    while chunk := stream.read(size):
        yield chunk


def _to_feature_row(game: ParsedGame, dataset_class: str) -> DatasetFeatureRow:
    headers = game.headers
    move_count = sum(1 for _ in game.pgn.split() if "." not in _)
    if move_count < 1:
        raise ValueError("game has no moves")
    return DatasetFeatureRow(
        dataset_class=dataset_class,
        game_id=game.lichess_game_id,
        username=headers.get("White") or headers.get("Black"),
        move_count=move_count,
        rated=headers.get("Event", "").casefold().startswith("rated"),
        speed=headers.get("Speed"),
        result=headers.get("Result"),
    )


def _write_parquet_atomic(path: Path, rows: Iterable[DatasetFeatureRow]) -> dict[str, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("collect-dataset requires pyarrow for Parquet output") from exc
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist([row.model_dump() for row in rows])
    pq.write_table(table, temporary, compression="zstd", row_group_size=512)
    schema = pq.read_schema(temporary)
    if list(schema.names) != FEATURE_COLUMNS:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Parquet schema validation failed")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    size = temporary.stat().st_size
    os.replace(temporary, path)
    return {"path": str(path), "bytes": size, "sha256": digest, "columns": FEATURE_COLUMNS}


def _require_period(name: str, value: str | None) -> None:
    if not value:
        raise MissingCollectionPeriod(f"{name} must be provided before network access")
