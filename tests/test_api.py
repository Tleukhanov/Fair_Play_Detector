"""API regression tests using the local SQLite database."""

import asyncio
from collections.abc import Generator

import httpx
import pytest

from apps.api.app.main import app


@pytest.fixture
def api_client() -> Generator[httpx.AsyncClient, None, None]:
    async def create_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    client = asyncio.run(create_client())
    yield client
    asyncio.run(client.aclose())


def test_health_endpoint(api_client: httpx.AsyncClient) -> None:
    response = asyncio.run(api_client.get("/health"))
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
