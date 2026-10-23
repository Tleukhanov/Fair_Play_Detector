"""Provider tests with an injected httpx mock transport (no real network)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from packages.llm.provider import (
    GeminiProvider,
    InvalidJSONError,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderTimeout,
    parse_json_response,
)

Handler = Callable[[httpx.Request], httpx.Response]
ProviderFactory = Callable[[httpx.AsyncClient], object]
MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Explain."},
]


def _run(handler: Handler, factory: ProviderFactory) -> str:
    async def scenario() -> str:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = factory(client)
            return await provider.complete(MESSAGES)  # type: ignore[attr-defined]

    return asyncio.run(scenario())


def _openai(
    client: httpx.AsyncClient,
    *,
    max_retries: int = 1,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        base_url="https://example.invalid/v1",
        timeout_seconds=1.0,
        max_retries=max_retries,
        client=client,
    )


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ]
        },
    )


def test_openai_success_and_fenced_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["model"] == "test-model"
        assert request.headers["Authorization"] == "Bearer test-key"
        return _completion('```json\n{"summary": "ok"}\n```')

    result = _run(handler, _openai)

    assert result == '{"summary": "ok"}'
    assert parse_json_response(result) == {"summary": "ok"}


def test_retries_once_on_429_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return _completion('{"ok": true}')

    result = _run(handler, _openai)

    assert result == '{"ok": true}'
    assert calls["count"] == 2


def test_retries_once_on_500_then_raises() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "server"})

    with pytest.raises(ProviderError, match="HTTP 500"):
        _run(handler, _openai)
    assert calls["count"] == 2


def test_timeout_maps_to_provider_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow")

    with pytest.raises(ProviderTimeout):
        _run(handler, _openai)


def test_malformed_api_json_is_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not json",
            headers={"content-type": "text/plain"},
        )

    with pytest.raises(ProviderError, match="malformed API JSON"):
        _run(handler, _openai)


def test_malformed_model_text_is_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion("это не JSON")

    result = _run(handler, _openai)

    assert result == "это не JSON"
    with pytest.raises(InvalidJSONError):
        parse_json_response(result)


def test_gemini_success_parses_parts_and_sends_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "gemini-key"
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["systemInstruction"]["parts"][0]["text"] == "Return JSON."
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"summary": '}, {"text": '"ok"}'}]
                        }
                    }
                ]
            },
        )

    def factory(client: httpx.AsyncClient) -> GeminiProvider:
        return GeminiProvider(
            api_key="gemini-key",
            model="gemini-test",
            base_url="https://example.invalid/v1beta",
            timeout_seconds=1.0,
            client=client,
        )

    assert _run(handler, factory) == '{"summary": "ok"}'
