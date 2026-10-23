"""Grounded RAG explainer with a hard deterministic fallback.

The flow is: retrieve knowledge -> build prompt -> call provider -> parse JSON ->
validate -> one repair retry -> deterministic fallback.  No provider exception or
invalid output is allowed to escape :meth:`LLMExplainer.aexplain`: the numeric
verdict is always returned with an explanation, and the explanation can never
change the score.
"""

from __future__ import annotations

from typing import Protocol

import httpx
from pydantic import ValidationError

from .config import LLMConfig
from .contracts import ExplanationInput, ExplanationOutput
from .fallback import DeterministicExplainer
from .prompt import PROMPT_VERSION, build_messages, build_repair_messages
from .provider import (
    ChatProvider,
    GeminiProvider,
    InvalidJSONError,
    OpenAICompatibleProvider,
    ProviderTimeout,
    parse_json_response,
)
from .rag import (
    KnowledgeBase,
    build_query,
    load_knowledge,
    retrieve,
)
from .validator import ExplanationValidationError, validate_explanation


class Explainer(Protocol):
    """Common async interface for LLM and deterministic explainers."""

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        """Return an explanation; must not raise for provider failures."""
        ...


class LLMExplainer:
    """RAG-backed explainer over a strict JSON contract."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        provider: ChatProvider | None = None,
        knowledge: KnowledgeBase | None = None,
        client: httpx.AsyncClient | None = None,
        fallback: DeterministicExplainer | None = None,
    ) -> None:
        self._config = config
        self._knowledge = knowledge or load_knowledge(config.knowledge_path)
        self._provider = provider or _build_provider(config, client)
        self._fallback = fallback or DeterministicExplainer()

    async def aexplain(self, data: ExplanationInput) -> ExplanationOutput:
        if not self._config.api_key.strip():
            return self._fallback_output(data, "no_api_key")
        chunks = retrieve(self._knowledge, build_query(data))
        messages = build_messages(data, chunks)

        try:
            text = await self._provider.complete(messages)
        except ProviderTimeout:
            return self._fallback_output(data, "timeout")
        except Exception:  # noqa: BLE001 - no provider failure may escape
            return self._fallback_output(data, "provider_error")

        output, failure = self._parse_output(text, data)
        if output is not None:
            try:
                validate_explanation(output, data, chunks)
            except ExplanationValidationError as exc:
                failure = ("validation_failed", str(exc))
            else:
                return output

        reason, detail = failure
        repair_messages = build_repair_messages(messages, text, detail)
        try:
            repaired_text = await self._provider.complete(repair_messages)
        except ProviderTimeout:
            return self._fallback_output(data, "timeout")
        except Exception:  # noqa: BLE001 - no provider failure may escape
            return self._fallback_output(data, "provider_error")

        repaired, repaired_failure = self._parse_output(repaired_text, data)
        if repaired is not None:
            try:
                validate_explanation(repaired, data, chunks)
            except ExplanationValidationError:
                return self._fallback_output(data, "validation_failed")
            return repaired
        return self._fallback_output(data, repaired_failure[0] or reason)

    def _parse_output(
        self,
        text: str,
        data: ExplanationInput,
    ) -> tuple[ExplanationOutput | None, tuple[str, str]]:
        try:
            payload = parse_json_response(text)
        except InvalidJSONError as exc:
            return None, ("invalid_json", str(exc))

        # Fields owned by the pipeline, not by the model.
        payload["source"] = "llm"
        payload["provider"] = self._config.provider
        payload["model"] = self._config.model
        payload["prompt_version"] = PROMPT_VERSION
        payload["knowledge_version"] = self._knowledge.version
        payload["fallback_reason"] = None
        try:
            output = ExplanationOutput.model_validate(payload)
        except ValidationError as exc:
            return None, ("validation_failed", str(exc))
        if not output.summary.strip():
            return None, ("validation_failed", "summary must not be empty")
        return output, ("", "")

    def _fallback_output(
        self,
        data: ExplanationInput,
        reason: str,
    ) -> ExplanationOutput:
        output = self._fallback.explain(data)
        return output.model_copy(update={"fallback_reason": reason})


def build_explainer(
    config: LLMConfig,
    *,
    client: httpx.AsyncClient | None = None,
    knowledge: KnowledgeBase | None = None,
) -> Explainer:
    """Return an LLM explainer when configured, otherwise deterministic.

    A disabled feature, an empty API key, an unsupported provider, or an
    unreadable knowledge file all degrade to the deterministic explainer instead
    of failing a scan.
    """

    if not config.configured:
        return DeterministicExplainer()
    try:
        base = knowledge or load_knowledge(config.knowledge_path)
        provider = _build_provider(config, client)
    except (OSError, ValueError):
        return DeterministicExplainer()
    return LLMExplainer(
        config,
        provider=provider,
        knowledge=base,
        client=client,
    )


def _build_provider(
    config: LLMConfig,
    client: httpx.AsyncClient | None,
) -> ChatProvider:
    if config.provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleProvider(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            client=client,
        )
    if config.provider == "gemini":
        return GeminiProvider(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            client=client,
        )
    raise ValueError(f"unsupported LLM provider: {config.provider!r}")


__all__ = ["Explainer", "LLMExplainer", "build_explainer"]
