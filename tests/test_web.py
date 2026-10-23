"""Static frontend shell tests: same-origin /app mount and API coexistence."""

import asyncio
from collections.abc import Generator

import httpx
import pytest

from apps.api.app.main import app

JS_MODULES = [
    "js/app.js",
    "js/api.js",
    "js/components.js",
    "js/state.js",
    "js/format.js",
    "js/views/home.js",
    "js/views/scan.js",
    "js/views/result.js",
    "js/views/game.js",
    "js/views/methodology.js",
]


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


def test_app_shell_has_navigation_theme_and_footer(web_client: httpx.AsyncClient) -> None:
    response = asyncio.run(web_client.get("/app/"))
    html = response.text

    assert 'lang="ru"' in html
    assert 'class="skip-link"' in html
    assert 'aria-label="Основная навигация"' in html
    assert "data-theme-toggle" in html
    assert "data-nav=\"home\"" in html
    assert "data-nav=\"methodology\"" in html
    assert "<footer" in html
    assert 'rel="noopener noreferrer"' in html
    assert 'aria-live="polite"' in html


@pytest.mark.parametrize("module", JS_MODULES)
def test_all_js_modules_are_served(web_client: httpx.AsyncClient, module: str) -> None:
    response = asyncio.run(web_client.get(f"/app/{module}"))

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "export" in response.text


def test_stylesheet_has_reduced_motion_and_responsive_rules(
    web_client: httpx.AsyncClient,
) -> None:
    response = asyncio.run(web_client.get("/app/styles.css"))
    css = response.text

    assert "prefers-reduced-motion" in css
    assert "@media (max-width" in css
    assert "focus-visible" in css
    assert "color-scheme" in css


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/app/styles.css", "text/css"),
        ("/app/assets/logo.svg", "image/svg+xml"),
        ("/app/assets/favicon.svg", "image/svg+xml"),
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
