"""Build the grounded explanation engine from API settings.

The explainer factory is deliberately failure-tolerant: a disabled feature, an
empty API key, an unsupported provider, or an unreadable knowledge file must
never prevent the API from starting.  In all those cases the deterministic
explainer is returned and scans keep producing an explanation.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from packages.llm import (
    DeterministicExplainer,
    Explainer,
    LLMConfig,
    build_explainer,
)
from packages.llm.provider import DEFAULT_GEMINI_BASE_URL, DEFAULT_OPENAI_BASE_URL

from ..config import Settings

logger = logging.getLogger(__name__)


def create_explainer(settings: Settings) -> Explainer:
    """Return an LLM explainer when configured, else the deterministic one."""

    provider = settings.llm_provider.strip().casefold()
    base_url = settings.llm_base_url
    if provider == "gemini" and base_url.rstrip("/") == DEFAULT_OPENAI_BASE_URL:
        # The generic setting defaults to the OpenAI endpoint; switching the
        # provider alone should not send Gemini requests to the OpenAI host.
        base_url = DEFAULT_GEMINI_BASE_URL

    try:
        config = LLMConfig(
            enabled=settings.llm_enabled,
            provider=provider,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )
        knowledge_path = settings.llm_knowledge_path.strip()
        if knowledge_path:
            config = replace(config, knowledge_path=Path(knowledge_path))
    except ValueError as exc:
        logger.warning("LLM configuration is invalid, using deterministic fallback: %s", exc)
        return DeterministicExplainer()

    explainer = build_explainer(config)
    if isinstance(explainer, DeterministicExplainer):
        if not config.enabled:
            logger.info("LLM explanations disabled; using deterministic explainer")
        elif not config.api_key.strip():
            logger.info("LLM_API_KEY is not set; using deterministic explainer")
        else:
            logger.warning(
                "LLM explainer could not be initialized; using deterministic explainer"
            )
    else:
        logger.info(
            "LLM explainer enabled provider=%s model=%s", config.provider, config.model
        )
    return explainer


__all__ = ["create_explainer"]
