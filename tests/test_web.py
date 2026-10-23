"""Static frontend shell tests: same-origin /app mount and API coexistence."""

import asyncio
from collections.abc import Generator

import httpx
import pytest

from apps.api.app.main import app


@pytest.fixture
def web_client() -> Generator[httpx.AsyncClient, None, None]:
    async def create_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    client = asyncio.run(create_client())
    yield client
    asyncio.run(client.aclose())


def test_root_redirects_to_app(web_client: httpx.AsyncClient) -> None:
    response = asyncio.run(web_client.get("/"))

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/app/"


def test_app_shell_is_served(web_client: httpx.AsyncClient) -> None:
    response = asyncio.run(web_client.get("/app/"))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="scan-form"' in response.text
    assert "fair play" in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/app/styles.css", "text/css"),
        ("/app/js/app.js", "javascript"),
        ("/app/assets/logo.svg", "image/svg+xml"),
    ],
)
def test_static_assets_are_served(
    web_client: httpx.AsyncClient, path: str, content_type: str
) -> None:
    response = asyncio.run(web_client.get(path))

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.content


def test_api_is_not_shadowed(web_client: httpx.AsyncClient) -> None:
    health = asyncio.run(web_client.get("/health"))
    openapi = asyncio.run(web_client.get("/openapi.json"))

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert openapi.status_code == 200
    assert "/api/v1/scans" in openapi.json()["paths"]
