#!/usr/bin/env python
"""Download the pinned ChessFraud parquet tables from Hugging Face.

The script keeps raw data out of git, records the immutable dataset revision and
per-file SHA-256 checksums in ``manifest.json``, and skips files whose checksum
already matches the manifest.  Downloads are streamed to a temporary ``.part``
file so an interrupted run never leaves a half-written parquet behind.

Usage::

    python scripts/download_chessfraud.py
    python scripts/download_chessfraud.py --output-dir /data/chessfraud --force

Set ``HF_TOKEN`` in the environment to raise Hugging Face rate limits; the
script forwards it as a request header and never writes it to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

LOGGER = logging.getLogger("download_chessfraud")

DATASET_ID = "artemlepin/chess-fraud"
HF_ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "bd2804f268bf07c306217929db9b8dda5803392b"
MANIFEST_NAME = "manifest.json"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One parquet file in the dataset repository."""

    source: str
    name: str


FILES: tuple[RemoteFile, ...] = (
    RemoteFile("data/chess_fraud/full.parquet", "chessfraud_full.parquet"),
    RemoteFile("data/chess_fraud_synth/train.parquet", "synth_train.parquet"),
    RemoteFile("data/chess_fraud_synth/test.parquet", "synth_test.parquet"),
)

Manifest = dict[str, object]


def sha256_file(path: Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> Manifest:
    """Read a manifest, returning an empty dict when it is absent or invalid."""

    if not path.is_file():
        return {}
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("ignoring unreadable manifest %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("ignoring manifest %s: top-level value is not an object", path)
        return {}
    return {str(key): value for key, value in payload.items()}


def write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically write ``manifest`` as sorted, indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _entries_by_name(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        return {}
    entries: dict[str, Mapping[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str):
            entries[name] = item
    return entries


def _entry_matches(entry: Mapping[str, object], destination: Path, url: str) -> bool:
    """Return True when the local file matches a manifest entry and URL."""

    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    expected_url = entry.get("url")
    if not isinstance(expected_sha, str) or not isinstance(expected_url, str):
        return False
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        return False
    if expected_url != url or not destination.is_file():
        return False
    if destination.stat().st_size != expected_bytes:
        return False
    return sha256_file(destination) == expected_sha


def _resolve_revision(client: httpx.Client, revision: str) -> str:
    """Resolve a branch, tag, or commit to an immutable commit SHA."""

    encoded = quote(revision, safe="")
    url = f"{HF_ENDPOINT}/api/datasets/{DATASET_ID}/revision/{encoded}"
    response = client.get(url)
    if response.status_code == 404 and revision != "main":
        LOGGER.warning("revision %r does not exist; falling back to 'main'", revision)
        return _resolve_revision(client, "main")
    response.raise_for_status()
    payload: object = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected revision response from {url}")
    resolved = payload.get("sha")
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(f"revision response from {url} has no commit SHA")
    return resolved


def _file_url(revision: str, source: str) -> str:
    return f"{HF_ENDPOINT}/datasets/{DATASET_ID}/resolve/{revision}/{source}"


def _http_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _stream_download(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[int, str]:
    """Stream ``url`` into ``destination``; return ``(bytes, sha256)``."""

    temporary = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    total = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total, digest.hexdigest()


def download_dataset(
    *,
    output_dir: Path,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
) -> Manifest:
    """Download all tables and write ``<output_dir>/manifest.json``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    previous = read_manifest(manifest_path)
    previous_entries = _entries_by_name(previous)

    timeout = httpx.Timeout(60.0, read=120.0)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_http_headers()) as client:
        resolved_revision = _resolve_revision(client, revision)
        LOGGER.info("dataset %s revision %s", DATASET_ID, resolved_revision)

        entries: list[dict[str, object]] = []
        for remote in FILES:
            url = _file_url(resolved_revision, remote.source)
            destination = output_dir / remote.name
            if not force and _entry_matches(
                previous_entries.get(remote.name, {}), destination, url
            ):
                LOGGER.info("skip %s (sha256 matches manifest)", remote.name)
                entry = previous_entries[remote.name]
                entries.append(
                    {
                        "name": remote.name,
                        "source": remote.source,
                        "url": url,
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                    }
                )
                continue

            LOGGER.info("download %s -> %s", remote.source, destination)
            size, digest = _stream_download(client, url, destination)
            LOGGER.info("  %d bytes sha256=%s", size, digest)
            entries.append(
                {
                    "name": remote.name,
                    "source": remote.source,
                    "url": url,
                    "bytes": size,
                    "sha256": digest,
                }
            )

    manifest: Manifest = {
        "dataset": DATASET_ID,
        "requested_revision": revision,
        "revision": resolved_revision,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": entries,
    }
    write_manifest(manifest_path, manifest)
    LOGGER.info("wrote %s", manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_raw"),
        help="directory for parquet files and manifest.json (default: data_raw)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"dataset revision to download (default: pinned {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when checksums already match the manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        manifest = download_dataset(
            output_dir=args.output_dir,
            revision=args.revision,
            force=args.force,
        )
    except (httpx.HTTPError, OSError, RuntimeError) as exc:
        LOGGER.error("download failed: %s", exc)
        return 1

    files = _entries_by_name(manifest)
    for name in sorted(files):
        entry = files[name]
        LOGGER.info("%s %s %s", name, entry.get("bytes"), entry.get("sha256"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
