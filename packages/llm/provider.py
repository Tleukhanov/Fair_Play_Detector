"""Async LLM providers with one retry on 429/5xx.

Only two shapes are implemented because both are OpenAI-compatible enough for a
strict JSON response: the standard ``/chat/completions`` endpoint and Google
Gemini's ``generateContent``.  No provider is called at import time, and tests
inject an ``httpx.AsyncClient`` with a mock transport, so this module never needs
network access in the test suite.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import httpx

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class ProviderError(RuntimeError):
    """Base class for provider failures that should trigger a fallback."""


class ProviderTimeout(ProviderError):
    """Raised when the provider did not answer within the configured timeout."""


class InvalidJSONError(ProviderError):
    """Raised when the model text is not a parseable JSON object."""


class ChatProvider(Protocol):
    """Minimal async chat interface used by :class:`LLMExplainer`."""

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the raw model text for ``messages``."""
        ...


class _HTTPProvider:
    """Shared retry/client plumbing for concrete providers."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._timeout = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._client = client

    @asynccontextmanager
    async def _client_scope(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client is not None:
            yield self._client
            return
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            yield client

    async def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._client_scope() as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=body,
                        params=params,
                        timeout=self._timeout,
                    )
            except httpx.TimeoutException as exc:
                raise ProviderTimeout("provider request timed out") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"provider transport error: {exc}") from exc

            last_status = response.status_code
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self._max_retries:
                    continue
                raise ProviderError(f"provider returned HTTP {response.status_code}")
            if response.status_code >= 400:
                raise ProviderError(f"provider returned HTTP {response.status_code}")

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("provider returned malformed API JSON") from exc
            if not isinstance(payload, dict):
                raise ProviderError("provider returned a non-object JSON payload")
            return payload

        raise ProviderError(f"provider failed after retries (last HTTP {last_status})")


class OpenAICompatibleProvider(_HTTPProvider):
    """Client for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
        json_mode: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
        if not model:
            raise ValueError("model must not be empty")
        self._api_key = api_key
        self._model = model
        self._temperature = float(temperature)
        self._max_output_tokens = int(max_output_tokens)
        self._json_mode = json_mode
        self._url = _completion_url(base_url)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": False,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = await self._post_json(self._url, headers=headers, body=body)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response has no choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider response has empty content")
        return strip_json_fences(content)


class GeminiProvider(_HTTPProvider):
    """Client for Gemini ``generateContent`` with JSON mime output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
        if not model:
            raise ValueError("model must not be empty")
        self._api_key = api_key
        self._model = model
        self._temperature = float(temperature)
        self._max_output_tokens = int(max_output_tokens)
        self._url = f"{base_url.rstrip('/')}/models/{model}:generateContent"

    async def complete(self, messages: list[dict[str, str]]) -> str:
        system_parts = [
            message["content"] for message in messages if message.get("role") == "system"
        ]
        user_parts = [
            message["content"] for message in messages if message.get("role") != "system"
        ]
        if not user_parts:
            raise ProviderError("Gemini request needs a user message")
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "\n\n".join(user_parts)}],
                }
            ],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system_parts:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        headers = {"Content-Type": "application/json"}
        payload = await self._post_json(
            self._url,
            headers=headers,
            body=body,
            params={"key": self._api_key},
        )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError("Gemini response has no candidates")
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise ProviderError("Gemini response has no content parts")
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text.strip():
            raise ProviderError("Gemini response has empty text")
        return strip_json_fences(text)


def _completion_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def strip_json_fences(text: str) -> str:
    """Strip a surrounding Markdown JSON fence, if present."""

    match = _FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse model text into a JSON object, tolerating fences and prose.

    Raises :class:`InvalidJSONError` for anything that is not a JSON object.
    """

    cleaned = strip_json_fences(text)
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        raise InvalidJSONError("model JSON is not an object")
    raise InvalidJSONError("model response is not valid JSON")


__all__ = [
    "DEFAULT_GEMINI_BASE_URL",
    "DEFAULT_OPENAI_BASE_URL",
    "ChatProvider",
    "GeminiProvider",
    "InvalidJSONError",
    "OpenAICompatibleProvider",
    "ProviderError",
    "ProviderTimeout",
    "parse_json_response",
    "strip_json_fences",
]
