"""Grounded explanation layer: contracts, RAG, validator, and safe fallback."""

from .config import LLMConfig
from .contracts import (
    ExplanationDirection,
    ExplanationEvidence,
    ExplanationInput,
    ExplanationOutput,
    ExplanationReasoning,
    ExplanationSource,
    ExplanationThresholds,
)
from .explainer import Explainer, LLMExplainer, build_explainer
from .fallback import DeterministicExplainer
from .prompt import PROMPT_VERSION, build_messages
from .provider import (
    ChatProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderTimeout,
)
from .rag import (
    KnowledgeBase,
    KnowledgeChunk,
    build_query,
    load_knowledge,
    retrieve,
)
from .validator import ExplanationValidationError, validate_explanation

__all__ = [
    "ChatProvider",
    "DeterministicExplainer",
    "Explainer",
    "ExplanationDirection",
    "ExplanationEvidence",
    "ExplanationInput",
    "ExplanationOutput",
    "ExplanationReasoning",
    "ExplanationSource",
    "ExplanationThresholds",
    "ExplanationValidationError",
    "GeminiProvider",
    "KnowledgeBase",
    "KnowledgeChunk",
    "LLMConfig",
    "LLMExplainer",
    "OpenAICompatibleProvider",
    "PROMPT_VERSION",
    "ProviderError",
    "ProviderTimeout",
    "build_explainer",
    "build_messages",
    "build_query",
    "load_knowledge",
    "retrieve",
    "validate_explanation",
]
