"""Configuration for the explanation layer.

The defaults are deliberately safe: the feature is enabled, but without an API
key :func:`packages.llm.explainer.build_explainer` still returns the deterministic
explainer.  Nothing here reads a secret from a file that could be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .provider import DEFAULT_GEMINI_BASE_URL, DEFAULT_OPENAI_BASE_URL
from .rag import DEFAULT_KNOWLEDGE_PATH


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime settings for the LLM explainer."""

    enabled: bool = True
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout_seconds: float = 30.0
    max_retries: int = 1
    temperature: float = 0.2
    max_output_tokens: int = 900
    knowledge_path: Path = field(default_factory=lambda: DEFAULT_KNOWLEDGE_PATH)

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "openai_compatible", "gemini"}:
            raise ValueError(f"unsupported LLM provider: {self.provider!r}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    @property
    def configured(self) -> bool:
        """True only when the LLM path may actually call a provider."""

        return self.enabled and bool(self.api_key.strip())

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build config from ``LLM_*`` environment variables."""

        provider = os.environ.get("LLM_PROVIDER", "openai").strip().casefold()
        default_base = (
            DEFAULT_GEMINI_BASE_URL
            if provider == "gemini"
            else DEFAULT_OPENAI_BASE_URL
        )
        knowledge_raw = os.environ.get("LLM_KNOWLEDGE_PATH", "").strip()
        return cls(
            enabled=_env_bool("LLM_ENABLED", True),
            provider=provider,
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", default_base),
            timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 1),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 900),
            knowledge_path=(
                Path(knowledge_raw) if knowledge_raw else DEFAULT_KNOWLEDGE_PATH
            ),
        )


__all__ = ["LLMConfig"]
