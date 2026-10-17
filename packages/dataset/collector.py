"""Safe, bounded dataset collection with an explicit post-collection gate."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import os
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import httpx

from packages.ingest.pgn import ParsedGame

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
    lichess_url_template: str = (
        "https://database.nikonoel.fr/lichess_elite_{period}.zip"
    ),
    tcec_url_template: str = (
        "https://github.com/TCEC-Chess/tcecgames/releases/download/S28-final/"
        "TCEC-everything-compact.zip"
    ),
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
    reports: list[SourceCollectionReport] = []
    source_metadata: dict[str, dict[str, object]] = {}
    samples: list[DatasetFeatureRow] = []
    try:
        for dataset_class, source, period, template in (
            ("human", "lichess_elite", lichess_elite_month, lichess_url_template),
            ("engine_like", "tcec", tcec_season, tcec_url_template),
        ):
            url = template.format(period=period)
            report, metadata, source_samples = _collect_source(
                http_client,
                quota,
                dataset_class,
                source,
                period,
                url,
                output_dir,
            )
            reports.append(report)
            source_metadata[dataset_class] = metadata
            samples.extend(source_samples)
        if sum(item.games_processed for item in reports) < 5:
            raise ValueError("dataset must contain at least five feature rows for the gate report")
        counts = {
            "human": reports[0].games_processed,
            "engine_like": reports[1].games_processed,
        }
        total = sum(counts.values())
        report = DatasetCollectionReport(
            lichess_elite_month=lichess_elite_month,
            tcec_season=tcec_season,
            quota_bytes=quota.limit,
            downloaded_bytes=quota.used,
            parquet_files=source_metadata,
            sources=reports,
            class_counts=counts,
            class_percentages={
                key: round(value / total * 100, 2) for key, value in counts.items()
            },
            sample_rows=samples[:5],
            columns=FEATURE_COLUMNS,
        )
        (output_dir / "collection_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        if own_client:
            http_client.close()


def _collect_source(
    client: httpx.Client,
    quota: ByteQuota,
    dataset_class: str,
    source: str,
    period: str,
    url: str,
    output_dir: Path,
) -> tuple[SourceCollectionReport, dict[str, object], list[DatasetFeatureRow]]:
    source_start = quota.used
    with client.stream("GET", url, headers={"Accept": "application/octet-stream"}) as response:
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix or ".archive"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw:
            temporary = Path(raw.name)
            for chunk in response.iter_bytes():
                quota.consume(len(chunk))
                raw.write(chunk)
    parquet_path = output_dir / (
        f"{'lichess_elite_' if dataset_class == 'human' else 'tcec_'}{period}_features.parquet"
    )
    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    writer = None
    row_count = 0
    row_batch: list[DatasetFeatureRow] = []
    samples: list[DatasetFeatureRow] = []
    skip_reasons: Counter[str] = Counter()
    games_read = games_skipped = 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for game in _iter_archive_games(temporary):
            games_read += 1
            try:
                row = _to_feature_row(game, dataset_class)
                if len(samples) < 5:
                    samples.append(row)
                row_batch.append(row)
                row_count += 1
                if len(row_batch) >= 512:
                    writer = _write_row_batch(pa, pq, writer, temporary_parquet, row_batch)
                    row_batch.clear()
            except ValueError as exc:
                games_skipped += 1
                skip_reasons[str(exc)] += 1
        if writer is None:
            if not row_batch:
                raise ValueError("source produced no valid feature rows")
            writer = _write_row_batch(pa, pq, writer, temporary_parquet, row_batch)
        elif row_batch:
            writer = _write_row_batch(pa, pq, writer, temporary_parquet, row_batch)
        writer.close()
        schema = pq.read_schema(temporary_parquet)
        if list(schema.names) != FEATURE_COLUMNS:
            raise RuntimeError("Parquet schema validation failed")
        digest = hashlib.sha256(temporary_parquet.read_bytes()).hexdigest()
        size = temporary_parquet.stat().st_size
        os.replace(temporary_parquet, parquet_path)
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"dataset source format failure: {source}: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        temporary_parquet.unlink(missing_ok=True)
    return (
        SourceCollectionReport(
            source=source,
            period=period,
            url=url,
            downloaded_bytes=quota.used - source_start,
            games_read=games_read,
            games_processed=row_count,
            games_skipped=games_skipped,
            skip_reasons=dict(skip_reasons),
        ),
        {
            "path": str(parquet_path),
            "bytes": size,
            "sha256": digest,
            "columns": FEATURE_COLUMNS,
        },
        samples,
    )


def _write_row_batch(pa, pq, writer, path: Path, rows: list[DatasetFeatureRow]):
    table = pa.Table.from_pylist([row.model_dump() for row in rows])
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _iter_archive_games(path: Path) -> Iterator[ParsedGame]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            pgn_names = [name for name in archive.namelist() if name.lower().endswith(".pgn")]
            if not pgn_names:
                raise ValueError("zip archive contains no PGN")
            for name in pgn_names:
                with archive.open(name, "r") as stream:
                    yield from _iter_fast_pgn_games(_chunks(stream))
        return
    opener = _decompressor(path)
    with opener(path, "rb") as stream:
        yield from _iter_fast_pgn_games(_chunks(stream))


def _iter_fast_pgn_games(chunks: Iterator[bytes]) -> Iterator[ParsedGame]:
    """Extract bounded dataset metadata without full engine-grade PGN validation."""

    buffer = ""
    current: list[str] = []
    for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        lines = buffer.splitlines(keepends=True)
        buffer = lines.pop() if lines and not lines[-1].endswith(("\n", "\r")) else ""
        for line in lines:
            if line.startswith("[Event ") and current:
                parsed = _fast_parse_game(current)
                if parsed is not None:
                    yield parsed
                current = []
            current.append(line)
    if buffer:
        current.append(buffer)
    if current:
        parsed = _fast_parse_game(current)
        if parsed is not None:
            yield parsed


def _fast_parse_game(lines: list[str]) -> ParsedGame | None:
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for line in lines:
        match = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$', line.strip())
        if match:
            headers[match.group(1)] = match.group(2)
        elif line.strip():
            movetext.append(line.strip())
    if not headers or not movetext:
        return None
    headers.setdefault("GameId", headers.get("Site", "").rstrip("/").split("/")[-1])
    return ParsedGame(headers=headers, pgn="".join(lines))


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
    move_count = sum(1 for token in game.pgn.split() if token.rstrip(".").isdigit())
    if move_count < 1:
        raise ValueError("game has no moves")
    game_id = game.headers.get("GameId") or hashlib.sha256(game.pgn.encode()).hexdigest()[:20]
    return DatasetFeatureRow(
        dataset_class=dataset_class,
        game_id=game_id,
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
