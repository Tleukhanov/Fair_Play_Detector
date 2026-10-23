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
