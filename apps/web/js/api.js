// Shared API client for the static frontend.
//
// The page is served from the same origin as FastAPI (`/app/`), so the default
// base is relative.  A separate dev server can override it with
// `<meta name="api-base">` or `window.FPD_API_BASE`.

const API_BASE =
  document.querySelector('meta[name="api-base"]')?.getAttribute("content") ||
  window.FPD_API_BASE ||
  "/api/v1";

const TERMINAL_STATUSES = new Set(["completed", "failed"]);
const DEFAULT_TIMEOUT_MS = 20000;
// Retrying these cannot help: the scan or game does not exist for this id.
const FATAL_STATUSES = new Set([404, 410]);

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function detailMessage(detail, fallback) {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
          return [loc, item.msg].filter(Boolean).join(": ");
        }
        return "";
      })
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }
  return fallback || "Запрос не удался";
}

async function request(
  path,
  { method = "GET", body, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {},
) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const controller = new AbortController();
  let timedOut = false;
  const timer =
    timeoutMs > 0
      ? setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeoutMs)
      : null;
  const forwardAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", forwardAbort, { once: true });
  }

  let response;
  let text = "";
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    text = await response.text();
  } catch (error) {
    if (signal?.aborted) throw error;
    if (timedOut) {
      throw new ApiError("Сервер не ответил вовремя. Проверьте подключение и повторите.", {
        status: 0,
        detail: error,
      });
    }
    if (error?.name === "AbortError") throw error;
    throw new ApiError("Сервер недоступен. Проверьте, что API запущен.", {
      status: 0,
      detail: error,
    });
  } finally {
    if (timer !== null) clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", forwardAbort);
  }

  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const detail = payload?.detail ?? payload ?? response.statusText;
    throw new ApiError(detailMessage(detail, response.statusText), {
      status: response.status,
      detail,
    });
  }
  return payload;
}

export function createScan({ username, maxGames, speed, ratedOnly }) {
  return request("/scans", {
    method: "POST",
    body: {
      username,
      max_games: maxGames,
      speed,
      rated_only: ratedOnly,
    },
  });
}

export function getScan(scanId, { signal } = {}) {
  return request(`/scans/${encodeURIComponent(scanId)}`, { signal });
}

export function getVerdict(scanId, { signal } = {}) {
  return request(`/scans/${encodeURIComponent(scanId)}/verdict`, { signal });
}

export function getGames(scanId, { page = 1, pageSize = 20, signal } = {}) {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  return request(`/scans/${encodeURIComponent(scanId)}/games?${params}`, { signal });
}

export function getGame(scanId, gameId, { signal } = {}) {
  return request(
    `/scans/${encodeURIComponent(scanId)}/games/${encodeURIComponent(gameId)}`,
    { signal },
  );
}

export function isTerminal(status) {
  return TERMINAL_STATUSES.has(status);
}

/**
 * Poll one scan until it reaches `completed`/`failed` or `stop()` is called.
 *
 * Calls are sequential: a new request is only scheduled after the previous one
 * settles, so a slow API cannot produce overlapping status requests.
 */
export function pollScan(
  scanId,
  { onUpdate, onError, intervalMs = 1500, maxNetworkErrors = 5 } = {},
) {
  let stopped = false;
  let timer = null;
  let networkErrors = 0;

  const stop = () => {
    stopped = true;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const schedule = (delay) => {
    if (stopped) return;
    timer = setTimeout(tick, delay);
  };

  async function tick() {
    if (stopped) return;
    try {
      const status = await getScan(scanId);
      if (stopped) return;
      networkErrors = 0;
      onUpdate?.(status);
      if (isTerminal(status.status)) {
        stop();
        return;
      }
      schedule(intervalMs);
    } catch (error) {
      if (stopped) return;
      if (error?.status && FATAL_STATUSES.has(error.status)) {
        stop();
        onError?.(error);
        return;
      }
      networkErrors += 1;
      if (networkErrors > maxNetworkErrors) {
        stop();
        onError?.(error);
        return;
      }
      schedule(intervalMs);
    }
  }

  tick();
  return stop;
}

export { API_BASE };
