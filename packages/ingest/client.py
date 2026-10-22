"""Async streaming client for the public Lichess game export."""

import asyncio
from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from time import time

import httpx

from .username import normalize_username


#: Lichess rejects the default httpx user agent with HTTP 404.  A descriptive
#: agent (and the public project URL as contact) is required for API access.
USER_AGENT = "fair-play-detector/0.1 (+https://github.com/Tleukhanov/Fair_Play_Detector)"


class LichessClientError(RuntimeError):
    """Raised when Lichess cannot provide a valid export response."""


class RateLimitError(LichessClientError):
    """Raised after the bounded retry budget is exhausted for HTTP 429."""


class LichessClient:
    """Fetch public rated games without buffering the complete PGN response."""

    def __init__(
        self,
        base_url: str = "https://lichess.org",
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        user_agent: str = USER_AGENT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not user_agent.strip():
            raise ValueError("user_agent must not be empty")
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._user_agent = user_agent
        self._client = client

    async def iter_pgn_chunks(
        self,
        username: str,
        *,
        max_games: int = 30,
        speed: str = "blitz",
        rated_only: bool = True,
    ) -> AsyncIterator[bytes]:
        """Yield response chunks from the official user export endpoint."""

        username = normalize_username(username)
        if not 1 <= max_games <= 50:
            raise ValueError("max_games must be between 1 and 50")
        if speed not in {"blitz", "rapid"}:
            raise ValueError("speed must be blitz or rapid")

        params = {
            "max": str(max_games),
            "perfType": speed,
            "rated": "1" if rated_only else "0",
            # Required for %clk annotations, i.e. per-move thinking time.
            "clocks": "true",
        }
        url = f"{self._base_url}/api/games/user/{username}"
        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            for attempt in range(self._max_retries + 1):
                try:
                    async with client.stream(
                        "GET",
                        url,
                        params=params,
                        headers={
                            "Accept": "application/x-chess-pgn",
                            "User-Agent": self._user_agent,
                        },
                    ) as response:
                        if response.status_code == 429:
                            if attempt >= self._max_retries:
                                raise RateLimitError("Lichess rate limit retry budget exhausted")
                            await asyncio.sleep(self._retry_delay(response, attempt))
                            continue
                        if response.status_code >= 500:
                            if attempt >= self._max_retries:
                                raise LichessClientError(
                                    f"Lichess server error: HTTP {response.status_code}"
                                )
                            await asyncio.sleep(self._retry_delay(response, attempt))
                            continue
                        if response.status_code >= 400:
                            raise LichessClientError(
                                f"Lichess export failed: HTTP {response.status_code}"
                            )
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                        return
                except httpx.TimeoutException as exc:
                    if attempt >= self._max_retries:
                        raise LichessClientError("Lichess export timed out") from exc
                    await asyncio.sleep(self._retry_delay(None, attempt))
                except httpx.NetworkError as exc:
                    if attempt >= self._max_retries:
                        raise LichessClientError("Lichess export network failure") from exc
                    await asyncio.sleep(self._retry_delay(None, attempt))
        finally:
            if own_client:
                await client.aclose()

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        return max(0.0, parsedate_to_datetime(retry_after).timestamp() - time())
                    except (TypeError, ValueError, OverflowError):
                        pass
        return float(2**attempt)
