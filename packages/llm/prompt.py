"""Prompt construction for the grounded RAG explainer.

The prompt is deliberately boring: policy chunks verbatim, an explicit JSON
output contract, and a compact payload containing only the numeric verdict and
already-computed evidence.  No identity, PGN, game ID, or free-form user text is
included because :class:`~packages.llm.contracts.ExplanationInput` cannot carry
it.
"""

from __future__ import annotations

import json
from typing import Any

from .contracts import ExplanationInput
from .rag import KnowledgeChunk

#: Version of the prompt template, stored with every LLM explanation.
PROMPT_VERSION = "rag-v1"

_HARD_RULES = """\
Жёсткие правила:
1. Не обвиняй игрока. Запрещены слова и корни «читер-», «винов-»,
«доказывает», «доказано», «обман-», «жулик-», «наказан-», «санкц-»,
«cheat-», «guilty», «proven», «ban», «sanction». Слово «доказательство»
разрешено только в отрицании «не является доказательством».
2. Используй только числа и факты из входного JSON и приложенных фрагментов
знаний. Если факта нет — напиши, что данных недостаточно. Не придумывай
статистику, метрики, проценты, имена, турниры и ссылки.
3. Объясняй, почему именно числовая модель приняла такое решение: опирайся
на evidence, contribution, contribution_share и direction. Не пересчитывай
и не меняй риск-скор.
4. Ссылайся только на источники из списка knowledge_chunks, указывая их id
и title. Обязательны источники policy.no_accusation и policy.factual_only.
5. Пиши по-русски, нейтрально, без канцелярита. Summary — 3–5 предложений.
Reasoning — 3–5 пунктов, каждый привязан к одному evidence_code из входных
данных. Limitations — не пустой список.
6. В summary обязательно должна быть фраза «не является доказательством».
7. Учитывай доверительный интервал и размер выборки: при широком интервале
или малом числе партий скажи об этом.
8. Не упоминай ник, имя, рейтинг, нотацию партии, идентификаторы партий
или соперников. Говори «игрок».

Формат ответа — строгий JSON без markdown-обёртки, ровно такие поля:
{
  "summary": "3–5 предложений по-русски",
  "reasoning": [
    {
      "evidence_code": "код из входного evidence",
      "observation": "что показывает число",
      "interpretation": "почему это сдвинуло решение модели",
      "direction": "supports_risk | reduces_risk | neutral",
      "contribution": 0.0
    }
  ],
  "sources": [{"id": "policy.no_accusation", "title": "Запрет обвинений"}],
  "limitations": ["ограничение из входных данных или knowledge_chunks"]
}
Не добавляй поля source, provider, model, prompt_version, knowledge_version:
их заполнит вызывающий код. Не добавляй никаких других полей.
"""


def build_messages(
    data: ExplanationInput,
    chunks: list[KnowledgeChunk],
) -> list[dict[str, str]]:
    """Build OpenAI-compatible ``system`` + ``user`` chat messages."""

    policy_chunks = [chunk for chunk in chunks if chunk.id.startswith("policy.")]
    if not policy_chunks:
        raise ValueError("prompt requires at least one policy chunk")

    policy_text = "\n\n".join(
        f"[{chunk.id}] {chunk.title}\n{chunk.text}" for chunk in policy_chunks
    )
    system = (
        "Ты объясняешь решение численной анти-читерской модели Fair Play Detector. "
        "Твоя задача — причинное объяснение в терминах модели, а не новое решение.\n\n"
        f"Политики (обязательны, дословно):\n{policy_text}\n\n"
        f"{_HARD_RULES}"
    )

    payload: dict[str, Any] = {
        "risk_level": data.risk_level,
        "risk_score": data.risk_score,
        "confidence_interval": list(data.confidence_interval),
        "games_analyzed": data.games_analyzed,
        "thresholds": (
            data.thresholds.model_dump() if data.thresholds is not None else None
        ),
        "model_version": data.model_version,
        "limitations": list(data.limitations),
        "evidence": [
            item.model_dump(exclude_none=True) for item in data.evidence
        ],
        "knowledge_chunks": [
            {"id": chunk.id, "title": chunk.title, "text": chunk.text}
            for chunk in chunks
        ],
    }
    user = (
        "Входные данные вердикта и источники:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nВерни только JSON по заданной схеме."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    messages: list[dict[str, str]],
    previous_response: str,
    error: str,
) -> list[dict[str, str]]:
    """Append a single repair turn after a schema or validation failure."""

    return [
        *messages,
        {"role": "assistant", "content": previous_response[:4000]},
        {
            "role": "user",
            "content": (
                "Предыдущий ответ отклонён: "
                f"{error}. Исправь только это и верни строгий JSON по схеме. "
                "Не добавляй пояснений вне JSON."
            ),
        },
    ]


__all__ = ["PROMPT_VERSION", "build_messages", "build_repair_messages"]
