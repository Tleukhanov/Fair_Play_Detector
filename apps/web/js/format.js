// Pure presentation helpers for the result and game views.
//
// Nothing here imports the API client: callers pass already-fetched values.
// Every helper tolerates null/undefined and returns an em dash instead of
// throwing or rendering "NaN"/"Invalid Date".

const DASH = "—";

function toNumber(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/** Risk score with three decimals, e.g. `0.359`. */
export function formatScore(value) {
  const number = toNumber(value);
  return number === null ? DASH : number.toFixed(3);
}

/** Share/probability as a percentage, e.g. `0.359` -> `36%`. */
export function formatPercent(value, digits = 0) {
  const number = toNumber(value);
  if (number === null) return DASH;
  const places = Number.isInteger(digits) && digits >= 0 ? digits : 0;
  return `${(number * 100).toFixed(places)}%`;
}

/** Confidence interval, e.g. `[0.2958, 0.4268]` -> `0.296–0.427`. */
export function formatCi(value) {
  if (!Array.isArray(value) || value.length < 2) return DASH;
  const low = toNumber(value[0]);
  const high = toNumber(value[1]);
  if (low === null || high === null) return DASH;
  return `${low.toFixed(3)}–${high.toFixed(3)}`;
}

/** Centipawn loss rounded to an integer, e.g. `24.6` -> `25 cp`. */
export function formatCpl(value) {
  const number = toNumber(value);
  return number === null ? DASH : `${Math.round(number)} cp`;
}

/** API timestamps are naive UTC; render them in the browser's local time. */
export function formatDateTimeUtc(value) {
  if (value === null || value === undefined || value === "") return DASH;
  let text;
  if (value instanceof Date) {
    text = value;
  } else if (typeof value === "string") {
    text = value.trim();
  } else {
    return DASH;
  }
  if (text === "") return DASH;
  let candidate = text;
  if (typeof candidate === "string" && !/(?:z|[+-]\d{2}:?\d{2})$/i.test(candidate)) {
    candidate = `${candidate}Z`;
  }
  const date = candidate instanceof Date ? candidate : new Date(candidate);
  if (Number.isNaN(date.getTime())) return DASH;
  const pad = (part) => String(part).padStart(2, "0");
  return (
    `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${date.getFullYear()} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`
  );
}

function normalizeColor(color) {
  if (typeof color !== "string") return null;
  const value = color.trim().toLowerCase();
  if (value === "white" || value === "w" || value === "белые") return "white";
  if (value === "black" || value === "b" || value === "чёрные" || value === "черные") {
    return "black";
  }
  return null;
}

/** Lichess result (`1-0`, `0-1`, `1/2-1/2`) from the scanned player's side. */
export function formatResult(result, userColor) {
  if (typeof result !== "string" || result.trim() === "") return DASH;
  const raw = result.trim();
  const color = normalizeColor(userColor);
  let winner = null;
  if (raw === "1-0") winner = "white";
  else if (raw === "0-1") winner = "black";
  else if (raw === "1/2-1/2" || raw === "½-½" || raw === "0.5-0.5") winner = "draw";
  if (winner === null) return raw;
  if (winner === "draw") return "Ничья";
  if (color === null) return raw;
  return winner === color ? "Победа" : "Поражение";
}

const RISK_LABELS = {
  low_risk: "Низкий риск",
  suspicious: "Подозрительно",
  high_risk: "Высокий риск",
};

const RISK_CLASSES = {
  low_risk: "low",
  suspicious: "suspicious",
  high_risk: "high",
};

export function riskLabel(level) {
  if (typeof level !== "string") return DASH;
  return RISK_LABELS[level] || level;
}

export function riskClass(level) {
  if (typeof level !== "string") return "neutral";
  return RISK_CLASSES[level] || "neutral";
}

const EVIDENCE_LABELS = {
  GAME_TOP_1_MATCH_RATE: "Совпадение с top-1",
  GAME_IDENTICAL_TIME_SHARE: "Одинаковые интервалы времени",
  GAME_MEAN_CENTIPAWN_LOSS: "Средний CPL",
  GAME_MEDIAN_CENTIPAWN_LOSS: "Медианный CPL",
  GAME_P90_CENTIPAWN_LOSS: "CPL 90-го перцентиля",
  GAME_TACTICAL_ERROR_COUNT: "Тактические ошибки",
  GAME_CENTIPAWN_LOSS_STABILITY: "Стабильность CPL",
  GAME_MEDIAN_TIME_SECONDS: "Медианное время на ход",
  GAME_MEAN_TIME_SECONDS: "Среднее время на ход",
  GAME_TIME_VARIANCE: "Разброс времени на ход",
  GAME_HAS_TIMING: "Наличие часов",
  HIGH_TOP_MOVE_RATE: "Высокая доля top-1",
  LOW_CENTIPAWN_LOSS: "Низкий средний CPL",
  REGULAR_MOVE_TIMES: "Регулярное время на ход",
  MAIA_DIVERGENCE: "Расхождение с Maia",
};

const RATE_CODES = new Set([
  "GAME_TOP_1_MATCH_RATE",
  "GAME_IDENTICAL_TIME_SHARE",
  "HIGH_TOP_MOVE_RATE",
  "REGULAR_MOVE_TIMES",
]);

const TIME_CODES = new Set([
  "GAME_MEDIAN_TIME_SECONDS",
  "GAME_MEAN_TIME_SECONDS",
]);

const CPL_CODES = new Set([
  "GAME_MEAN_CENTIPAWN_LOSS",
  "GAME_MEDIAN_CENTIPAWN_LOSS",
  "GAME_P90_CENTIPAWN_LOSS",
  "LOW_CENTIPAWN_LOSS",
]);

const COUNT_CODES = new Set(["GAME_TACTICAL_ERROR_COUNT"]);

function upperCode(code) {
  return typeof code === "string" ? code.trim().toUpperCase() : "";
}

/** Russian label for a backend evidence code, else a readable fallback. */
export function evidenceLabel(code, fallback) {
  const key = upperCode(code);
  if (key && EVIDENCE_LABELS[key]) return EVIDENCE_LABELS[key];
  if (typeof fallback === "string" && fallback.trim() !== "") {
    return fallback.trim();
  }
  return humanizeCode(code);
}

/** Humanize an UPPER_SNAKE code, e.g. `GAME_TOP_1_MATCH_RATE` -> `Top 1 match rate`. */
export function humanizeCode(code) {
  if (typeof code !== "string" || code.trim() === "") return DASH;
  const cleaned = code
    .trim()
    .replace(/^GAME_/i, "")
    .replace(/[_\s]+/g, " ")
    .toLowerCase();
  if (cleaned === "") return DASH;
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

/** Format an evidence value with the unit implied by its code. */
export function formatEvidenceValue(code, value) {
  const number = toNumber(value);
  if (number === null) return DASH;
  const key = upperCode(code);
  if (RATE_CODES.has(key)) return formatPercent(number, 0);
  if (TIME_CODES.has(key)) return `${number.toFixed(1)} с`;
  if (key === "GAME_TIME_VARIANCE") return `${number.toFixed(2)} с²`;
  if (CPL_CODES.has(key)) return formatCpl(number);
  if (COUNT_CODES.has(key)) return String(Math.round(number));
  if (key === "GAME_HAS_TIMING") return number > 0 ? "есть" : "нет";
  return number.toFixed(2);
}

/** Cohort baseline uses exactly the same units as the observed value. */
export function formatEvidenceBaseline(code, value) {
  return formatEvidenceValue(code, value);
}

/** Evidence weight, e.g. `0.19`. */
export function formatWeight(value) {
  const number = toNumber(value);
  return number === null ? DASH : number.toFixed(2);
}

// ---------- LLM explanation helpers ----------
//
// The backend already decided whether an explanation came from a provider or
// from the deterministic fallback. These helpers only label that decision; they
// never infer anything about the player or change a number.

const EXPLANATION_SOURCE_LABELS = {
  llm: "LLM",
  deterministic: "без LLM",
};

export function explanationSourceLabel(source) {
  if (typeof source !== "string") return DASH;
  return EXPLANATION_SOURCE_LABELS[source.trim().toLowerCase()] || DASH;
}

const DIRECTION_LABELS = {
  supports_risk: "повышает риск",
  reduces_risk: "снижает риск",
  neutral: "нейтрально",
};

export function directionLabel(direction) {
  if (typeof direction !== "string") return DIRECTION_LABELS.neutral;
  return DIRECTION_LABELS[direction.trim().toLowerCase()] || DIRECTION_LABELS.neutral;
}

const DIRECTION_CLASSES = {
  supports_risk: "direction-up",
  reduces_risk: "direction-down",
  neutral: "direction-neutral",
};

export function directionClass(direction) {
  if (typeof direction !== "string") return DIRECTION_CLASSES.neutral;
  return DIRECTION_CLASSES[direction.trim().toLowerCase()] || DIRECTION_CLASSES.neutral;
}

/** Signed model-space contribution, e.g. `+0.081` or `−0.042` (U+2212 minus). */
export function formatContribution(value) {
  const number = toNumber(value);
  if (number === null) return DASH;
  if (number === 0) return "0.000";
  const magnitude = Math.abs(number).toFixed(3);
  return number > 0 ? `+${magnitude}` : `\u2212${magnitude}`;
}

/** Share of total absolute attribution, e.g. `18%`. */
export function formatContributionShare(value) {
  const number = toNumber(value);
  return number === null ? DASH : `${Math.round(number * 100)}%`;
}

/** Retrieved source title, falling back to its id when the title is missing. */
export function sourceTitle(source) {
  const title = typeof source?.title === "string" ? source.title.trim() : "";
  if (title !== "") return title;
  const id = typeof source?.id === "string" ? source.id.trim() : "";
  return id !== "" ? id : DASH;
}

const FALLBACK_REASON_LABELS = {
  no_api_key: "LLM не настроен",
  provider_error: "Ошибка провайдера",
  timeout: "Таймаут провайдера",
  invalid_json: "Некорректный ответ LLM",
  validation_failed: "Ответ не прошёл проверку",
  unexpected_error: "Внутренняя ошибка",
};

/** Human label for a backend fallback code; unknown codes stay silent. */
export function fallbackReasonLabel(reason) {
  if (typeof reason !== "string") return "";
  return FALLBACK_REASON_LABELS[reason.trim().toLowerCase()] || "";
}
