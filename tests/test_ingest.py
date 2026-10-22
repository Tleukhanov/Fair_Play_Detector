"""Mocked contract tests for public Lichess ingestion."""

import asyncio

import httpx
import pytest

from packages.ingest import (
    LichessClient,
    LichessClientError,
    RateLimitError,
    iter_pgn_games,
    normalize_username,
)

PGN = """[Event "rated"]
[Site "https://lichess.org/abc123"]
[GameId "abc123"]
[Variant "Standard"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def test_ingest_streams_chunks_and_sends_export_filters() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=PGN.encode(),
            headers={"Content-Type": "application/x-chess-pgn"},
        )

    async def scenario() -> bytes:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [
                chunk
                async for chunk in LichessClient(client=client).iter_pgn_chunks(
                    "  Test_User ",
                    max_games=12,
                    speed="rapid",
                )
            ]
        return b"".join(chunks)

    assert asyncio.run(scenario()) == PGN.encode()
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url).endswith(
        "/api/games/user/test_user?max=12&perfType=rapid&rated=1&clocks=true"
    )
    assert request.headers["Accept"] == "application/x-chess-pgn"
    # Lichess returns 404 for the default httpx user agent; a descriptive one is
    # required for live API access.
    assert request.headers["User-Agent"].startswith("fair-play-detector/")


def test_ingest_retries_rate_limit_and_honors_retry_after() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, content=PGN.encode())

    async def scenario() -> bytes:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return b"".join(
                [
                    chunk
                    async for chunk in LichessClient(
                        client=client, max_retries=1, backoff_seconds=0
                    ).iter_pgn_chunks("tester")
                ]
            )

    assert asyncio.run(scenario()) == PGN.encode()
    assert attempts == 2


def test_ingest_exposes_exhausted_rate_limit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RateLimitError):
                async for _chunk in LichessClient(
                    client=client, max_retries=1, backoff_seconds=0
                ).iter_pgn_chunks("tester"):
                    pass

    asyncio.run(scenario())


def test_ingest_surfaces_http_errors_and_isolates_bad_pgn() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(LichessClientError, match="HTTP 404"):
                async for _chunk in LichessClient(client=client).iter_pgn_chunks("tester"):
                    pass

    asyncio.run(scenario())
    errors: list[str] = []
    games = list(iter_pgn_games((PGN + "\n[Event \"bad\"]\n\n").encode().splitlines(True), errors))
    assert [game.lichess_game_id for game in games] == ["abc123"]
    assert errors == ["PGN contains no moves"]


def test_ingest_normalizes_and_rejects_usernames() -> None:
    assert normalize_username("  Test_User ") == "test_user"
    with pytest.raises(ValueError, match="invalid Lichess username"):
        normalize_username("not valid")
